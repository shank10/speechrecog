#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 22:33:36 2018

@author: sshekhar
"""
import sys
import numpy as np
import librosa
import math
from keras.utils import Sequence
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import base64
import os
sample_audio_file = "sample.wav"
def encoded_audio(audio_file):
     audio_data = base64.encodebytes(open(audio_file,"rb").read())
     audio_string = audio_data.decode()
     return audio_string

def decode_audio(audio_string):
     audio = base64.b64decode(audio_string)
     return (audio)

def detect_speech(audio_string):
     if os.path.isfile(sample_audio_file):
          os.remove(sample_audio_file)
     audio = decode_audio(audio_string)
     open(sample_audio_file,"wb").write(audio)
     return

def spect_loader(path, window_size, window_stride, window, normalize, max_len=101, 
                 augment=False, allow_speedandpitch=False, allow_pitch=False,
                 allow_speed=False, allow_dyn=False, allow_noise=False,
                allow_timeshift=False ):
    y, sr = librosa.load(path, sr=None)
    n_fft = int(sr * window_size)
    win_length = n_fft
    hop_length = int(sr * window_stride)

    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    spect, phase = librosa.magphase(D)

    # S = log(S+1)
    spect = np.log1p(spect)

    # make all spects with the same dims
    # TODO: change that in the future
    if spect.shape[1] < max_len:
        pad = np.zeros((spect.shape[0], max_len - spect.shape[1]))
        spect = np.hstack((spect, pad))
    elif spect.shape[1] > max_len:
        spect = spect[:max_len, ]
    spect = np.resize(spect, (1, spect.shape[0], spect.shape[1]))
    #spect = torch.FloatTensor(spect)

    # z-score normalization
    if normalize:
        mean = np.mean(np.ravel(spect))
        std = np.std(np.ravel(spect))
        if std != 0:
            spect = spect -mean
            spect = spect / std

    return spect


def loadAndSpect(fname,  window_size, window_stride, window_type, normalize, max_len):
    img = spect_loader(fname, 
                       window_size, 
                       window_stride, 
                       window_type, 
                       normalize, 
                       max_len)
    img=np.swapaxes(img, 0, 2)

    x = img_to_array(img, data_format='channels_last')
    return x
            
class WavSequence(Sequence):

    def __init__(self, x_set, batch_size=64, window_size=0.02, window_stride=0.01, window_type='hamming', normalize=True, max_len=101):
        self.x = x_set
        self.batch_size = batch_size
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_type = window_type
        self.normalize = normalize
        self.max_len = max_len

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([
            loadAndSpect(file_name, window_size, window_stride, window_type, normalize, max_len)
               for file_name in batch_x])

#Delete these two lines at the time of Simpala integration
audio_file = sys.argv[1]
encoded_audio_string = encoded_audio(audio_file)
if len(encoded_audio_string)<10:
     print("Please call the model with a properly base64 encoded audio. The audio received with this request is corrupt")
     exit(1)     
detect_speech(encoded_audio_string)

window_size=.02
window_stride=.01
window_type='hamming'
normalize=True
max_len=101
batch_size = 1
sample_filenames=[sample_audio_file]
classnames=['stop', 'no', 'down', 'yes', 'up', 'silence', 'off', 'right', 'on', 'left', 'go', 'unknown']

seq = WavSequence(sample_filenames, batch_size=batch_size)
model = load_model('audio_cnn.model')
preds = model.predict_generator(generator=seq, 
                        steps=len(seq), 
                        workers=1, 
                        use_multiprocessing=False, 
                        verbose=0)

inv_map = {0: 'down', 1: 'go', 2: 'left', 3: 'no', 4: 'off', 5: 'on', 6: 'right', 7: 'silence', 8: 'stop', 9: 'unknown', 10: 'up', 11: 'yes'}
classes = np.argmax(preds, axis=1)
probes = np.max(preds, axis=1)

res = []
for cl in classes:
    res.append(inv_map[cl])
    
print("The speech command is: ",res)