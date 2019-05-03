from music21 import converter, instrument, note, chord, midi, stream
import numpy as np
import glob
import keras
import tensorflow as tf
from keras.utils import np_utils
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Input, LSTM, Dropout, Activation
#
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import sys
import random

notes = []

for i, file in enumerate(glob.glob("I:\\Programming\\AI\\Music Generator\\WindowsFormsApp1\\WindowsFormsApp1\\bin\\Debug\\Dataset\\" + sys.argv[1] + "/*.mid")):
    print(i)
    print("Music Readed")
    music = converter.parse(file)
    music = music[0]
    musicToNote = None
    musicToNote = music.flat.notes
    for elm in musicToNote:
        if isinstance(elm, note.Note):
            notes.append(str(elm.pitch))
        elif isinstance(elm, chord.Chord):
            notes.append('.'.join(str(n) for n in elm.normalOrder))
    
print("Hey I Loaded All Of Musics, I Am Ready!! :)")

notesName = sorted(set(elm for elm in notes))

notesNameSize = len(notesName)
notesSize = len(notes)

notesDic = dict()
for i, elm in enumerate(notesName):
    notesDic[elm] = i

seqLen = 50
trainingNum = notesSize - seqLen

data = np.zeros((trainingNum, seqLen, notesNameSize))
label = np.zeros((trainingNum, notesNameSize))

print(label)

for i in range(0, trainingNum):
    inputSeq = notes[i: i + seqLen]
    tmpLabel = notes[i + seqLen]
    for j, elm in enumerate(inputSeq):
        data[i][j][notesDic[elm]] = 1
    label[i][notesDic[tmpLabel]] = 1 

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(seqLen, notesNameSize)))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(notesNameSize))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

model.fit(data, label, batch_size=128, nb_epoch=100, shuffle=True)

model.save_weights("Weights\\" + sys.argv[1] + ".h5")