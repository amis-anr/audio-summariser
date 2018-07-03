<<<<<<< HEAD
import os,time

import numpy as np

from sklearn.externals       import joblib
from sklearn.linear_model    import LinearRegression as linreg
from sklearn.model_selection import train_test_split

from audio_summariser.audio_file    import AudioFile
from audio_summariser.audio_segment import AudioSegment
from audio_summariser import utils
from audio_summariser import summary_similarity as sumsim

class InfoLinearRegression:

  def __init__(self,X=None,Y=None,verbose='true'):
    self.X = X
    self.Y = Y

    self.model = None

    self.verbose = verbose

  # For convenience, WAV files and CTM files must have the exact same name.
  #
  def prepare_features(self,audio_corpus,ctm_dir):
    _x = []
    _y = []

    k = 0
    start = time.time()
    
    files = os.listdir(audio_corpus)

    if self.verbose:
      print('Starting feature preparation... '+str(len(files))+' files to process.')

    for f in files:
      fname, fext = os.path.splitext(f)
      file = os.path.join(audio_corpus,fname+'.wav')

      try:
        af = AudioFile(file)
      except audioop.error:
        continue

      af.compute_mfccs(mode='full')
  
      try:
        t_b = utils.bounded_transcript(os.path.join(ctm_dir,fname+'.ctm'))
      except (IndexError, FileNotFoundError):
        continue

      _ = [ sumsim.compute_sim(
          utils.load_full_transcript(os.path.join(ctm_dir,fname+'.ctm')),
          t['text']) 
       for t in t_b]

      for i in range(0,len(t_b)):
        if not t_b[i]['text'] or (t_b[i]['end']-t_b[i]['start']) < 5:
            continue
        s = round(t_b[i]['start']*af.sr/512)

        segment = AudioSegment(af.mfccs[...,s:s+431],s/float(af.mfccs.shape[1]),i,10)
        feats = None
        try:
          feats   = segment.compute_feature_stats()
        except ValueError:
          continue

        _x.append(feats)

        # We are training using the Jensen-Shannon Divergence as target metric.
        _y.append(_[i][1])

      k += 1
      if k%20 == 0 and self.verbose:
        print(' ')
        print('\t ___ '+str(time.time()-start)+' seconds elpased. '+str(k)+' files processed, '+str(len(files)-k)+' to go.')
      else:
        print('|', sep='', end='', flush=True)

    self.X = np.stack(_x, axis=0)
    self.Y = np.stack(_y, axis=0)


  def save(self):
    joblib.dump(self.X,'amis-linear-regression-10s-X.pkl')
    joblib.dump(self.Y,'amis-linear-regression-10s-Y.pkl')


  def train(self):
    _l = linreg()
    self.model = _l.fit(self.X,self.Y)

  def predict(self,x):
    return self.model.predict(x)

  @staticmethod
  def load_feature_dataset():
    # TODO: define a configuration file here to manage paths
    X = joblib.load('amis-linear-regression-10s-X.pkl')
    Y = joblib.load('amis-linear-regression-10s-Y.pkl')

    return X,Y

  @staticmethod
  def test_model(X,Y):
    _l = linreg()
    _ = []

    for i in range(0,10):
      X_train, X_test, y_train, y_test = train_test_split(
          #X,Y, test_size=0.3)
          X,preprocessing.MinMaxScaler((0,1)).fit_transform(Y.reshape(-1,1)), test_size=0.2 )
      model = _l.fit(X_train,y_train)#lab_enc.fit_transform(Y))
      predictions = model.predict(X_test)
      _.append(model.score(X_test,y_test))

    print('Averaged model score over 10 iterations: '+str(sum(_)/10))
