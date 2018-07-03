import math
import numpy as np

# This function takes the path of a CTM file as an argument, and extracts
# only the transcribed words.
#
def load_full_transcript(path):
  return ' '.join([ line.split()[4] for line in open(path,'r') ])

def load_summ_transcript(path):
  return ' '.join([ line.split('\t')[4] for line in open(path,'r')  if line.strip() ] )

# This function might be a bit redundant with load_full_transcript(), although
# it extracts all the information contained in the CTM file.
#
def parse_ctm(path):
    word_times = {}
    for line in open(path,'r'):
        sline = line.split()
        start_time = float(sline[2])
        word_times[start_time] = {}
        word_times[start_time]['duration'] = sline[3]#= str(float(sline[2])+float(sline[3]))
        word_times[start_time]['word']  = sline[4]
        word_times[start_time]['confidence']  = float(sline[5])

    return word_times

def bounded_transcript(path):#,bound_times_mfccs):
    transcript_time_words = parse_ctm(path)
    transcript_start_times = np.sort(list(transcript_time_words.keys()))
    
    last_word_time = list(transcript_time_words.keys())[-1]
    transcript_str = []
    transcript_b = []
    #for i in range(0,math.floor(last_word_time+float(transcript_time_words[last_word_time]['duration'])))
    for i in range(0,math.floor(last_word_time)):
        _ = []
        #transcript_idx = (np.abs(transcript_start_times - bound_times_mfccs[i])).argmin()
        transcript_idx = (np.abs(transcript_start_times - i)).argmin()
        
        for j in range(transcript_idx,len(transcript_start_times)):
            if (transcript_start_times[i] >= i+10) or (i+10-transcript_start_times[j]) < 0.1:
                break
                
            word = transcript_time_words[transcript_start_times[j]]['word']
            
            if word.upper() != "<UNK>":
                _.append(transcript_time_words[transcript_start_times[j]]['word'])
                
        #transcript_str.append(' '.join(_))
        transcript_b.append({'start': i, 'end': i+10, 'text': ' '.join(_) })

        
    return transcript_b
