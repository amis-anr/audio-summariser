import os,errno
import numpy as np
import librosa

from audio_summariser.audio_segment import AudioSegment

class AudioFile:

  def __init__(self,path):
    if not os.path.isfile(path):
      raise FileNotFoundError(
          errno.ENOENT, os.strerror(errno.ENOENT), filename)

    y, self.sr = librosa.load(path)
    self.y_mono = librosa.to_mono(y)  # since we don't take into account both the audio channels,
                                      # we'll always process the files as mono.
    self.length = librosa.get_duration(y)


    self.audio_segments = []
    self.mfccs = None
    self.bounds_mfcc = None
    self.bound_times_mfccs = None


  def segment_file(self,mode='background'):
    S = None

    if mode == 'background':
      S,p = self.get_background_audio()
    else:
      S,p = librosa.magphase(librosa.stft(self.y_mono))

    self.mfccs = librosa.feature.mfcc(S=S, sr=self.sr, n_fft=512, hop_length=256, n_mfcc=25)
    self.bounds_mfcc = librosa.segment.agglomerative(self.mfccs, round(librosa.get_duration(y=self.y_mono,sr=self.sr)*20/60))
    self.bound_times_mfccs = librosa.frames_to_time(self.bounds_mfcc, sr=self.sr) # extraction of the delimitations

  def get_background_audio(self):
    S_full, phase = librosa.magphase(librosa.stft(self.y_mono))

    # We'll compare frames using cosine similarity, and aggregate similar frames
    # by taking their (per-frequency) median value.
    #
    # To avoid being biased by local continuity, we constrain similar frames to be
    # separated by at least 2 seconds.
    #
    # This suppresses sparse/non-repetetitive deviations from the average spectrum,
    # and works well to discard vocal elements.

    S_filter = librosa.decompose.nn_filter(S_full,
                                           aggregate=np.median,
                                           metric='cosine',
                                           width=int(librosa.time_to_frames(2, sr=self.sr)))

    # The output of the filter shouldn't be greater than the input
    # if we assume signals are additive.  Taking the pointwise minimium
    # with the input spectrum forces this.
    S_filter = np.minimum(S_full, S_filter)

    # We can also use a margin to reduce bleed between the vocals and instrumentation masks.
    # Note: the margins need not be equal for foreground and background separation
    margin_i = 2
    power = 2

    mask_i = librosa.util.softmask(S_filter,
                                   margin_i * (S_full - S_filter),
                                   power=power)

    # Once we have the masks, simply multiply them with the input spectrum
    # to separate the components
    S_background = mask_i * S_full

    return S_background,phase
