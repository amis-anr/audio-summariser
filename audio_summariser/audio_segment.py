import librosa
import numpy as np
import scipy.stats as ss

class AudioSegment:

  def __init__(self,v_mfccs,start_frame):
    self.mfcc_mat = v_mfccs
    self.start    = start_frame

  def compute_feature_stats(self):#S):
    delta_mfccs  = librosa.feature.delta(self.mfcc_mat,mode='nearest') 
    delta_mfccs2 = librosa.feature.delta(self.mfcc_mat,order=2,mode='nearest')
    
    metric_min      = np.min( self.mfcc_mat, axis=1)
    metric_max      = np.max( self.mfcc_mat, axis=1)
    metric_median   = np.median( self.mfcc_mat, axis=1)
    metric_mean     = np.mean( self.mfcc_mat, axis=1)
    metric_variance = np.var( self.mfcc_mat, axis=1)
    metric_skewness = ss.skew( self.mfcc_mat, axis=1)
    metric_kurtosis = ss.kurtosis( self.mfcc_mat, axis=1)
    mean_d  = np.mean(delta_mfccs, axis=1)
    var_d   = np.var(delta_mfccs, axis=1)
    mean_d2 = np.mean(delta_mfccs2, axis=1)
    var_d2   = np.var(delta_mfccs2, axis=1)
    length = self.mfcc_mat.shape[1]
    
    
    return np.hstack((metric_min, metric_max, metric_median,metric_mean,
                      metric_variance,metric_skewness,metric_kurtosis,mean_d,mean_d2,var_d,var_d2,length,self.start)) 
