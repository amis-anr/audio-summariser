import math
import numpy as np

from audio_summariser.audio_file import AudioFile
from audio_summariser.audio_segment import AudioSegment
from audio_summariser.informativity_regression import InfoLinearRegression

class Summariser:

  def __init__(self,mode='ranking'):
    self.mode = mode

    if mode == 'ranking':
      self.linreg = InfoLinearRegression.load_trained_model()

  def summarise_file(self,audio_file):
    y_sum = None

    if self.mode == 'ranking':
      y_sum = self.summarise_rank(audio_file)
    elif self.mode == 'heuristic':
      y_sum = self.summarise_heuristic(audio_file)
      
    return y_sum

  def summarise_rank(self,audio_file,percentage=0.2):
    
    if not audio_file.bound_times_mfccs:
      audio_file.segment_file()
    
    _segments = []
    for i in range(0,len(audio_file.bound_times_mfccs)-1):
      start_time = math.ceil(audio_file.bound_times_mfccs[i])
      length_ratio = (audio_file.bound_times_mfccs[i+1]-audio_file.bound_times_mfccs[i])/audio_file.bound_times_mfccs[-1]

      s = round(start_time*audio_file.sr/512)
      segment = AudioSegment(audio_file.mfccs[...,s:s+431],s/float(audio_file.mfccs.shape[1]),i,audio_file.bound_times_mfccs[i+1]-audio_file.bound_times_mfccs[i])
      feats   = segment.compute_feature_stats()

      info_pred = self.linreg.predict(feats.reshape(1,-1))

      segment.compute_score(audio_file.bound_times_mfccs[i],audio_file.bound_times_mfccs[i+1],length_ratio,info_pred[0],start_time)

      _segments.append(segment)


    r_segs = sorted(_segments, key=lambda x: x.score, reverse=True)
    summ_l = 0
    summ_idx = []
    j = 0

    while summ_l/audio_file.length < percentage:
      summ_idx.append(r_segs[j].index)
      summ_l += r_segs[j].length
      j += 1

    selected_frames = [ audio_file.y_mono[int(round(audio_file.sr*audio_file.bound_times_mfccs[i])):int(round(audio_file.sr*audio_file.bound_times_mfccs[i+1]))] for i in sorted(summ_idx) ]

    y_sum = np.concatenate(selected_frames)

    return y_sum

