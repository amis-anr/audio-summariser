from audio_summariser.audio_file import AudioFile
from audio_summariser.summariser import Summariser

a = AudioFile("../wav/Euronews_fra_A_GHn7RkzPFvY_LItalie-accueille-Meriam-la-Soudanaise-qui-a-chapp--la-peine-de-mort.wav")
s = Summariser()
s.summarise_file(a)
