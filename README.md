# AI-LSTM_StoryGenerator

A research project for my Artificial Intelligence class from Fall 2020.  Here, we looked more into how Long Short Term Memory can be used to enhance text generation by implementing a multi-lay bidirectional LSTM.

I used Jason Brownlee's and David Campion's tutoritals for this project by combing the two's work.  I recommend checking out their tutorials as well to get a better understanding of their indivual projects since they implemted theirs differently.  These links will be cited at the bottom.
 
The tools I used when making this AI was PyCharm, Python 3.8, and Tensorflow.
 
Since I only used PyCharm for this research project I will explain how to run it within the PyCharm enviroment - however, this **might be different for you** if you plan to run it on your local machine or in a different enviroment.

# Example outputs

```
temp = 0.35

Seed:
"  i was lucky enough to invent a machine, which has "
2020-12-06 11:16:38.779035: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)

tily of the strange and stones.

now, and the train is the charm in the steps, and the stone of the proper of the cank of a shought of the roar which had been a little and tried to be an engine-dase of the 
stranger, and the boys that had been a little and tree.  the boys bross the shore of the boat and first prepared to the train to a short pair of the train.  the train came back 
to the train to the stock of the street. the brighter was so his face of the strange and stoned to the rest of the trains.  i am exarded that the boys was a little can in the 
boat and pretty and stone and the store of the books and path stroked the strange of the strange of the trains.

‘i think it is the bank of the strange trains,’ said anthea.

‘yes, it’s all the recond water of the trairs,’ that the door was as the part of the tunnel, and the train and the strange of the bank of the strange trees that the servant was 
a black pain of the street of the street, and it was a stone stroned the rest of the sunsh
```

```
temp - 0.2

Seed:
" ng, because i have read books myself, and i know h "
2020-12-06 11:02:58.767334: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)

ow and the stone of the strange and stone to the boat with a stranger, and the strange of the part of the children had not tried to be an engine-dase that the bottom of the 
street of the strange of the cank and strange of the strange of the strange and the street of the street of the strange breath of the strange and the strange of the strange 
trains of the hand, and the children had been a little of the strange and stone and tried to the strange of the street, and the train and the strange of the train and the 
children that it was a little of the street of the front of the strange travis and pretty and stroned, and the children had been all the strange travis that the strange was a 
little of the train and the children stood to the strange of the boat to the bottom of the train to the strange of the boat to the strange of the brass and the train to the 
strange and the princess of the princess of the sunnel of the train and the strange of the coat and the strange of the strange of the s
```

# Set Up

First, go ahead and download this GitHub, you should get all the information, from the code we used to some of our testing data.  Unzip this file and open the file in PyCharm.
Make sure that your PyCharm is in Python 3.8 and go to SETTINS (CTRL+ALT+S) and go to Python Intupter. On the right hand there should be a "+" icon - click that and search for the following:
  1. tensorflow
  2. tensorflow-datasets
  3. keras
  4. pillow
  
 These should be all the data you would need to run the code within PyCharm.
   - NOTE: If you run into problems with downloading any of this components, PyCharm should be able to tell you.  You can also change you Python version here depending on if you have it installed on your computer.
  
 Now, go to the train.py within PyCharm.  Here is where we train our data.  If you want to use our data that is fine!  It was all open source found at: 
   https://www.gutenberg.org/
   
 If you don't want to use the same data, go to "storyData.txt" delete what we currently have and replace with the data that you want. You can use just about any text to generate your own story.  The AI does need data to run!
 
 # Trainning Settings
 
 You can skip this step if you plan to use the data that I have already made.  You will have to go into the epoch20Testing file and drag out the "epoch-16-loss-1.2612.hdf5" into the overal file to get the write.py to work correctly.  Feel free to play around with any other tests that I have.
 
 To personalize this AI you can change some the data on certain lines.
 The lines that you can edit within "train.py" are 27, 51, 52, and 65.  
 
   - Keep in mind that some of these changes will also have to be applied to write.py
 
   - Line 28 --> the sequence length: 
     ```
     seqLength = 50
     ```
  
   - Line 51 --> relates to the how many hidden layers that will be in you LSTM (this is the 256):
     ```
     model.add(Bidirectional(LSTM(256, return_sequences=True, activation-"relu"), input_shape=(X.shpe[1], X.shape[2])))
     ```
  
   - Line 52 --> should have the same hidden layers this should have the same number in 51, this relates to how many rnn will be in you LSTM:
     ```
     model.add(Bidirectional(LSTM(256)))
     ```
  
   - Line 65 --> you can change the batch size and/or the epochs size, batch size looks at certain sequences of text while epochs runs those batch sizes:
     ```
     model.fit(X,y, batch_size=64, shuffle=True, epochs=30, callbacks=callbacks_list, validation_split=0.1
     ```
   
When everything is changed, you can run some tests by right clicking and clicking "run".  Please note, that with the current settings it took my computer 4 hours to run one epochs- **this code takes a lot of time**. And depending one your settings you might have to change other settings to keep a good test going.   You can lower some of the numbers (for example rnn) to speed up the process.  Just note that some of the data you will get from those tests will not be as accurate.

- NOTE: Depending on your system, you might get red errors.  If the program keeps running you should be fine and see some of these outputs after a while.

```
2020-12-11 11:27:57.478376: W tensorflow/stream_executor/platform/default/dso_loader.cc:60] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2020-12-11 11:27:57.480738: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
Total Characters:  1783326
Total Vocab:  58
Total Patterns:  1783276
2020-12-11 11:29:07.071727: I tensorflow/compiler/jit/xla_cpu_device.cc:41] Not creating XLA devices, tf_xla_enable_xla_devices not set
2020-12-11 11:29:07.076290: W tensorflow/stream_executor/platform/default/dso_loader.cc:60] Could not load dynamic library 'nvcuda.dll'; dlerror: nvcuda.dll not found
2020-12-11 11:29:07.077746: W tensorflow/stream_executor/cuda/cuda_driver.cc:326] failed call to cuInit: UNKNOWN ERROR (303)
2020-12-11 11:29:07.095899: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:169] retrieving CUDA diagnostic information for host: YOUR-COMPUTER NAME
2020-12-11 11:29:07.098283: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:176] hostname: YOUR-COMPUTER-NAME
2020-12-11 11:29:07.115172: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2020-12-11 11:29:07.123912: I tensorflow/compiler/jit/xla_gpu_device.cc:99] Not creating XLA devices, tf_xla_enable_xla_devices not set
2020-12-11 11:29:09.983012: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
Epoch 1/30
    5/25078 [..............................] - ETA: 4:22:59 - loss: 3.9758 - categorical_accuracy: 0.1106
```

I recommend coming back and checking on the test.  Sometimes you might run into an error where the loss becomes: nan and the accuracy is exteremely low.  These tests are awful and won't produce well written stores.  I recommend to stop your tests and then change up a few settings.  Don't be discourage if you do run into these problems though!  It takes a bit to understand what your data wants from you and only testing can get you to the correct place!

While this is running it will be saving files, its a good idea to keep track of which one holds both the best loss and accurancy while the code runs and pick that one for when we move onto the write file.

You'll know when the trainning has finished when the terminal outputs 
```
Move on to write.py
```

# Write you story!

The only changes you have to make here are the changes you made to your LSTM model, how long of a story you want it to print out, and the data file you want it to use.  This file is what your computer saves after every test and it's a good idea to use the lowest loss for the writing portion.

**If you made any changes in the train.py file you might have to change a few things in the write.py file as well**
You can also change a few things up in the write.py file to change your own output:

   - line 26 this seqLength **must** match the line 28 in train.py
     ```
     seqLength = 50
     ```
   - line 49 the RNN size **must** match the line 51 in train.py
     ```
     model.add(Bidirectional(LSTM(256, return_sequences=True, activation-"relu"), input_shape=(X.shpe[1], X.shape[2])))
     ```
   - line 50 the RNN size **must** match the line 52 in train.py
     ```
     model.add(Bidirectional(LSTM(256)))
     ```
   - line 55 this file **must** match the file **YOU** want to use for your story generation
     ```
     filename = "epoch-16-loss-1.2612.hdf5"
     ```
   - line 69 temperature can be changed based off of how random you want your story to be
     ```
     def sample(preds, temperature=0.25
     ```
   - line 80 the 1000 can be changed to how long you want your story to be
     ```
     for i in range(1000)
     ```

Keep playing around with all the settings and all the data files you have and see what kind of stories you can output!

# Recourses
### CODING RESOURSES:

Brownlee, J., 2020. Text Generation With LSTM Recurrent Neural Networks In Python With Keras. [online] Machine Learning Mastery. Available at: https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/ [Accessed 11 December 2020].

Campion, D., 2018. Text Generation Using Bidirectional LSTM And Doc2vec Models 1/3. [online] Medium. Available at: https://medium.com/@david.campion/text-generation-using-bidirectional-lstm-and-doc2vec-models-1-3-8979eb65cb3a [Accessed 11 December 2020].

### DATA RESOURCES:

Alger Jr, H., 2006. Ragged Dick;, By Horatio Alger Jr.. [online] Gutenberg.org. Available at: <https://www.gutenberg.org/files/5348/5348-h/5348-h.htm> [Accessed 11 December 2020].

Ballantyne, R., 2007. The Coral Island, By R.M. Ballantyne. [online] Gutenberg.org. Available at: <https://www.gutenberg.org/files/21721/21721-h/21721-h.htm> [Accessed 11 December 2020].

Ballantyne, R., 2007. The Coral Island. [online] Gutenberg.org. Available at: <https://www.gutenberg.org/files/646/646-h/646-h.htm> [Accessed 11 December 2020].

Nesbit, E., 2016. The Story Of The Amulet, By E. Nesbit. [online] Gutenberg.org. Available at: <https://www.gutenberg.org/files/837/837-h/837-h.htm> [Accessed 11 December 2020].

Nesbit, E., 2016. The Story Of The Treasure Seekers, By E. Nesbit. [online] Gutenberg.org. Available at: <https://www.gutenberg.org/files/770/770-h/770-h.htm> [Accessed 11 December 2020].

Nesbit, E., 2018. The Railway Children, By E. Nesbit. [online] Gutenberg.org. Available at: <https://www.gutenberg.org/files/1874/1874-h/1874-h.htm> [Accessed 11 December 2020].
