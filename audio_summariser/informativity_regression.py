from sklearn.externals    import joblib
from sklearn.linear_model import LinearRegression as linreg

class InfoLinearRegression:

  def __init__(self):
    self.X = None
    self.Y = None

  @staticmethod
  def load_trained_model():
    _l = linreg()
    X,Y = InfoLinearRegression.load_feature_dataset()
    model = _l.fit(X,Y)

    return model

  @staticmethod
  def load_feature_dataset():
    # TODO: define a configuration file here to manage paths
    X = joblib.load('amis-linear-regression-10s-X.pkl')
    Y = joblib.load('amis-linear-regression-10s-Y.pkl')

    return X,Y
