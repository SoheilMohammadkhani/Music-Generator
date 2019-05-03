    # The Libreries

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

for i, file in enumerate(glob.glob("I:\\Programming\\AI\\Music Generator\\WindowsFormsApp1\\WindowsFormsApp1\\bin\\Debug\\Dataset\\" + sys.argv[2] + "/*.mid")):
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

# model.fit(data, label, batch_size=128, nb_epoch=100, shuffle=True)

# model.save_weights("Weights\Persian.h5")

model.load_weights("I:\\Programming\\AI\\Music Generator\\WindowsFormsApp1\\WindowsFormsApp1\\bin\\Debug\\Weights\\" + sys.argv[2] + ".h5")

rDic = dict()
for elm in notesDic.keys():
    index = notesDic[elm]
    rDic[index] = elm

startNum = np.random.randint(0, len(data) - 1)
seq = data[startNum]
startSeq = seq.reshape(1, seqLen, notesNameSize)
output = []

for i in range(0, int(sys.argv[3])):
    newNote = model.predict(startSeq, verbose=0)
    ind = np.argmax(newNote)
    tNote = np.zeros((notesNameSize))
    tNote[ind] = 1
    output.append(tNote)
    seq = startSeq[0][1:]
    startSeq = np.concatenate((seq, tNote.reshape(1, notesNameSize)))
    startSeq = startSeq.reshape(1, seqLen, notesNameSize)

musicNotes = []
for elm in output:
    ind = list(elm).index(1)
    musicNotes.append(rDic[ind])

time = 0.0
musicNotes2 = []

for elm in musicNotes:
    if ('.' in elm) or elm.isdigit():
        chordNotes = elm.split('.')
        notes = []
        for tmpNote in chordNotes:
            newNote = note.Note(int(tmpNote))
            newNote.storedInstrument = instrument.Piano()
            notes.append(newNote)
        tmpChord = chord.Chord(notes)
        tmpChord.offset = time
        musicNotes2.append(tmpChord)
    else:
        newNote = note.Note(elm)
        newNote.offset = time
        newNote.storedInstrument = instrument.Piano()
        musicNotes2.append(newNote)
    time += 0.4

tmpStream = stream.Stream(musicNotes2)

path = "I:\\Programming\\AI\Music Generator\\WindowsFormsApp1\\WindowsFormsApp1\\bin\\Debug\\Musics\\" + sys.argv[1] + ".mid"

tmpStream.write('midi', fp=path)

print("Music Generated")