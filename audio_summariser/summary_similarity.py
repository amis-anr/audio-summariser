import math
import numpy as np

from nltk          import PorterStemmer
from nltk.corpus   import stopwords
from nltk.tokenize import word_tokenize

Stopwords = {
  'eng' : set(stopwords.words('english')),
  'fra' : set(stopwords.words('french')),
  'ara' : set(stopwords.words('arabic'))
}
delta = 0.0005


def kl(inpt,summ):
  _ = 0.0

  vnorm = np.unique(inpt).shape[0]

  for w in np.sort(np.unique(inpt)):
    pp = (inpt.count(w) + delta)/(len(inpt) + delta*1.5*vnorm)
    pq = (summ.count(w) + delta)/(len(summ) + delta*1.5*vnorm)
    
    _ += pp*math.log2(pp/pq)

  return _

def js(inpt,summ):
  _p = 0.0
  _q = 0.0

  vnorm = np.unique(inpt).shape[0]

  for w in np.sort(np.unique(inpt)):
    pp = (inpt.count(w) + delta)/(len(inpt) + delta*1.5*vnorm)
    pq = (summ.count(w) + delta)/(len(summ) + delta*1.5*vnorm)
    
    _p += pp*math.log2(pp/((pp+pq)/2))
    _q += pq*math.log2(pq/((pp+pq)/2))

  return (_p + _q)/2


def compute_sim(full_transcript,summary,lang='eng'):
  p = PorterStemmer()

  inpt_filtered = [ p.stem(w) for w in word_tokenize(full_transcript) if not w in Stopwords[lang] ]
  summ_filtered = [ p.stem(w) for w in word_tokenize(summary) if not w in Stopwords[lang] ]

  return [ globals()[metric](inpt_filtered,summ_filtered) for metric in ['kl','js'] ]
