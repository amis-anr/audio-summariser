# audio-summariser

This library provides utilities that allow to build audio summaries of audio files.
This method has been primarily tested on news, and does not rely on any linguistic resource, which makes it effective on any language (although we tested it on english, french, and arabic).

The main process for building such summaries is (roughly) as follows:
1. load and segment the audio file using the awesome [librosa](https://librosa.github.io/) library,
2. for each segment, compute statistics based on [MFCC features](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/), in order to predict its informativeness,
3. rank each segment based on the estimated informativeness and other parameters,
4. concatenate the audio frames of the top ranked segments, up to a given percentage of the original file (default to 20%), which gives the audio summary.


You can view the [example notebook](https://github.com/amis-anr/audio-summariser/blob/master/example-audio-summary.ipynb) for an overview.

### Linear regression for predicting the informativity of audio parts

The summariser heavily relies on estimating the informativity of the segments.
We use a simple linear regression model trained on our entire corpus, although this could be upgraded by plugin in neural nets.

#### Prerequisites

We do not provide a fully trained model in this repository, you'll have to train it yourself on your own data.
What you need is:
* a corpus of audio files,
* the text transcription of these files, using the [CTM file format](http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/infmts.htm#ctm_fmt_name_0)

/!\ The audio files and the CTM files must have the same name (apart from their extension).

#### Training

Training the linear regression model is pretty simple, but computing the features can be very long (~approx. 1h for 200 files of our corpus).
```python
from audio_summariser.informativity_regression import InfoLinearRegression

_i = InfoLinearRegression()
_i.prepare_features(audio_files_dir,ctm_files_dir)
_i.save()
_i.train()
```

The `_i.save()` call saves the computed features in two files: `amis-linear-regression-10s-X.pkl` and `amis-linear-regression-10s-Y.pkl`.
When calling the summariser, these two files are loaded to train the linear regression model.


### Summarising an audio file

```python
from audio_summariser.audio_file import AudioFile
from audio_summariser.summariser import Summariser

a = AudioFile(path_to_audio_file)
s = Summariser()
y_sum = s.summarise_file(a)
```

`y_sum` contains the audio frames of the summary.

You can listen to the summary directly in Jupyter using the built in audio functions:
```python
import IPython.display as ipd
ipd.Audio(y_sum,rate=a.sr)
```
