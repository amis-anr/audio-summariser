import math

from audio_summariser.audio_file import AudioFile
from audio_summariser.audio_segment import AudioSegment


class Summariser:

  def __init__(self,mode='ranking'):
    self.mode = mode

  def summarise_file(self,audio_file):
    y_sum = None

    if self.mode == 'ranking':
      y_sum = self.summarise_rank(audio_file)
    elif self.mode == 'heuristic':
      y_sum = self.summarise_heuristic(audio_file)
      
    return y_sum

  def summarise_rank(self,audio_file):
    if not audio_file.bound_times_mfccs:
      audio_file.segment_file()
    
    _scores = []
    for i in range(0,len(audio_file.bound_times_mfccs)-1):
      start_time = math.ceil(audio_file.bound_times_mfccs[i])
      length_ratio = (audio_file.bound_times_mfccs[i+1]-audio_file.bound_times_mfccs[i])/audio_file.bound_times_mfccs[-1]

      s = round(start_time*audio_file.sr/512)
      segment = AudioSegment(audio_file.mfccs[...,s:s+431],s/float(audio_file.mfccs.shape[1]))
      feats   = segment.compute_feature_stats()
