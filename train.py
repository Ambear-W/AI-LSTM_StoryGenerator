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

# load ascii text and covert to lowercase

raw_text = open("storyData.txt", 'r', encoding='utf-8').read()
raw_text = raw_text.lower()

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))

# normalize
X = X / float(n_vocab)

# one hot encode the output variable
y = np_utils.to_categorical(dataY)

# define the LSTM model
# model = Sequential()
# model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(256))
# model.add(Dropout(0.2))
# model.add(Dense(y.shape[1], activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam')

# define the checkpoint
# filepath="weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
# callbacks_list = [checkpoint]

# fit the model
# model.fit(X, y, epochs=2, batch_size=64, callbacks=callbacks_list)

model = Sequential()
# 256 = size of rnn
model.add(Bidirectional(LSTM(256, activation="relu"), input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.add(Activation('softmax'))
# 0.001 = learning rate
optimizer = Adam(lr=0.001)
callbacks = [EarlyStopping(patience = 2, monitor='val)loss')]
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[categorical_accuracy])
model.summary()

filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
callbacks = [EarlyStopping(patience=4, monitor='val_loss'), ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')]

# fit the model
model.fit(X, y, batch_size=64, shuffle=True, epochs=2, callbacks=callbacks, validation_split=0.1)
