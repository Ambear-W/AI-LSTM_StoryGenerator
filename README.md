# AI-LSTM_StoryGenerator
A research project for my Artificial Intelligence class from Fall 2020.  Here, we looked more into how Long Short Term Memory can be used to enhance text generation by implementing a multi-lay bidirectional LSTM.

I used Jason Brownlee's and David Campion's tutoritals for this project by combing the two's work.  I recommend checking out their tutorials as well to get a better understanding of their indivual projects since they implemted theirs differently.  These links will be cited at the bottom.
 
The tools I used when making this AI was PyCharm, Python 3.8, and Tensorflow.
 
Since I only used PyCharm for this research project I will explain how to run it within the PyCharm enviroment - however, this might be different for you if you plan to run it on your local machine.

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
   
 If you don't want to use the same data go to "storyData.txt" delete what we currently have and replace with the data that you want. 
 
 The lines that you can edit within "train.py" are 27, 51, 52, and 65.
 
   - Line 27 --> the sequence length
  
   - Line 51 --> relates to the size of the rnn
  
   - Line 52 --> should have the same rnn size
  
   - Line 65 --> you can change the batch size and/or the epochs size
   
When everything is changed, you can run some tests by right clicking and clicking "run".  Please note, that with the current settings it took my computer 4 hours to run one epochs- this code takes a lot of time. And depending one your settings you might have to change other settings to keep a good test going.   You can lower some of the numbers (for example rnn) to speed up the process.  Just note that some of the data you will get from those tests will not be as accurate.

I recommend coming back and checking on the test.  Sometimes you might run into an error where the loss becomes: nan and the accuracy is exteremely low.  These tests are awful and won't produce well written stores.  I recommend to stop your tests and then change up a few settings.

Don't be discourage if you do run into these problems though!  It takes a bit to understand what your data wants from you and only testing can get you to the correct place!

While this is running it will be saving files, its a good idea to keep track of which one holds both the best loss and accurancy while the code runs and pick that one for when we move onto the write file.

The only changes you have to make here are the changes you made to your LSTM model, how long of a story you want it to print out, and the data file you want it to use.  This file is what your computer saves after every test and it's a good idea to use the lowest loss for the writing portion.

CODING RESOURSES:
Brownlee, J., 2020. Text Generation With LSTM Recurrent Neural Networks In Python With Keras. [online] Machine Learning Mastery. Available at: https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/ [Accessed 11 December 2020].

Campion, D., 2018. Text Generation Using Bidirectional LSTM And Doc2vec Models 1/3. [online] Medium. Available at: https://medium.com/@david.campion/text-generation-using-bidirectional-lstm-and-doc2vec-models-1-3-8979eb65cb3a [Accessed 11 December 2020].

DATA RESOURCES:
Alger Jr, H., 2006. Ragged Dick;, By Horatio Alger Jr.. [online] Gutenberg.org. Available at: <https://www.gutenberg.org/files/5348/5348-h/5348-h.htm> [Accessed 11 December 2020].

Ballantyne, R., 2007. The Coral Island, By R.M. Ballantyne. [online] Gutenberg.org. Available at: <https://www.gutenberg.org/files/21721/21721-h/21721-h.htm> [Accessed 11 December 2020].

Ballantyne, R., 2007. The Coral Island. [online] Gutenberg.org. Available at: <https://www.gutenberg.org/files/646/646-h/646-h.htm> [Accessed 11 December 2020].

Nesbit, E., 2016. The Story Of The Amulet, By E. Nesbit. [online] Gutenberg.org. Available at: <https://www.gutenberg.org/files/837/837-h/837-h.htm> [Accessed 11 December 2020].

Nesbit, E., 2016. The Story Of The Treasure Seekers, By E. Nesbit. [online] Gutenberg.org. Available at: <https://www.gutenberg.org/files/770/770-h/770-h.htm> [Accessed 11 December 2020].

Nesbit, E., 2018. The Railway Children, By E. Nesbit. [online] Gutenberg.org. Available at: <https://www.gutenberg.org/files/1874/1874-h/1874-h.htm> [Accessed 11 December 2020].
