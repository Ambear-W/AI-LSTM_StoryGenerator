import numpy
import sys
from keras.models import Sequential
from keras.layers import Dense
# Dropout, Activation
from keras.layers import LSTM, Bidirectional
# from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
# from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.metrics import categorical_accuracy

# load data/text and makes all lower case --> making your data lowercase
# helps reduce the time that it takes to train your code

rawText = open("storyData.txt", 'r', encoding='utf-8').read()
rawText = rawText.lower()

# sorts the characters then adds them into an dictionary where they are numbered
sortedChars = sorted(list(set(rawText)))
charDictionary = dict((char, i) for i, char in enumerate(sortedChars))
charDictionaryInt = dict((i, char) for i, char in enumerate(sortedChars))

n_chars = len(rawText)
n_vocab = len(sortedChars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

# prepare the dataset of input to output pairs encoded as integers
seqLength = 50
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
model.add(Bidirectional(LSTM(256, return_sequences=True, activation="relu"), input_shape=(X.shape[1], X.shape[2])))
model.add(Bidirectional(LSTM(256)))
model.add(Dense(y.shape[1], activation='softmax'))
callbacks = [EarlyStopping(patience=2, monitor='val)loss')]
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[categorical_accuracy])
# load the network weights
filename = "epoch-16-loss-1.2612.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
# pick a random seed
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("Seed:")
print("\"", ''.join([charDictionaryInt[value] for value in pattern]), "\"")


# this prevents the AI from causing the story to repeat itself as well as how predictable the text will be
# the higher the temperature the less predictable the outcome will be --> this means that spelling/grammar errors could be more common
# the lower the temperature the more predictable the outcome will be
def sample(preds, temperature=0.2):
	preds = numpy.asarray(preds).astype('float64')
	numpy.seterr(divide='ignore')
	preds = numpy.log(preds) / temperature
	exp_preds = numpy.exp(preds)
	preds = exp_preds / numpy.sum(exp_preds)
	preds = numpy.ravel(preds)
	probas = numpy.random.multinomial(1, preds, 1)
	return numpy.argmax(probas)


# generate characters
for i in range(1000):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = sample(prediction)
	result = charDictionaryInt[index]
	seq_in = [charDictionaryInt[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]

print("\nDone.")
