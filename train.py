# This code heavy used the methods that Jason Brownlee and David Campion wrote respectfully.  I combined the
# two to increase the effiency of the story generator by impletmenting a bi-lstm and using a char by char generator
# Larger LSTM Network
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM, Bidirectional
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.metrics import categorical_accuracy

# load data/text and makes all lower case --> making your data lowercase
# helps reduce the time that it takes to train your code

rawText = open("storyData.txt", 'r', encoding='utf-8').read()
rawText = rawText.lower()

# sorts the characters then adds them into an dictionary where they are numbered
sortedChars = sorted(list(set(rawText)))
charDictionary = dict((char, i) for i, char in enumerate(sortedChars))

n_chars = len(rawText)
n_vocab = len(sortedChars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

# prepare the dataset of input to output pairs encoded as integers
seqLength = 100
dataX = []
dataY = []
for i in range(0, n_chars - seqLength, 1):
	seq_in = rawText[i:i + seqLength]
	seq_out = rawText[i + seqLength]
	dataX.append([charDictionary[char] for char in seq_in])
	dataY.append(charDictionary[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seqLength, 1))

# normalize
X = X / float(n_vocab)

# one hot encode the output variable
y = np_utils.to_categorical(dataY)

# define the LSTM --> this makes it so that way we can get the testing done to start writing up our stories!
model = Sequential()
# size of rnn = 256 --> this doesnt really matter but tends to stay in the 200-300 range
model.add(Bidirectional(LSTM(256, activation="relu"), input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(256)))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
callbacks = [EarlyStopping(patience=2, monitor='val)loss')]
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[categorical_accuracy])

# fit the model
filepath = "{epoch:02d}-loss-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# model.fit(X, y, epochs=50, batch_size=64, callbacks=callbacks_list)

# callbacks = [EarlyStopping(patience=4, monitor='val_loss'), ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')]

# fit the model
model.fit(X, y, batch_size=64, shuffle=True, epochs=1, callbacks=callbacks_list, validation_split=0.1)

print("Move on to write.py")
