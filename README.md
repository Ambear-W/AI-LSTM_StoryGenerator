# AI-LSTM_StoryGenerator
A research project for my Artificial Intelligence class from Fall 2020.  Here, we looked more into how Long Short Term Memory can be used to enhance text generation by implementing a multi-lay bidirectional LSTM.
 
The tools I used when making this AI was PyCharm, Python 3.8, and Tensorflow.
 
Since I only used PyCharm for this research project I will explain how to run it within the PyCharm enviroment - however, this might be different for you if you plan to run it on your local machine.

First, go ahead and download this GitHub, you should get all the information, from the code we used to some of our testing data.  Unzip this file and open the file in PyCharm.
Make sure that your PyCharm is in Python 3.8 and go to SETTINS (CTRL+ALT+S) and go to Python Intupter. On the right hand there should be a "+" icon - click that and search for the following:
  1. tensorflow
  2. tensorflow-datasets
  3. keras
  4. pillow
  
 These should be all the data you would need to run the code within PyCharm.
   NOTE: If you run into problems with downloading any of this components, PyCharm should be able to tell you.  You can also change you Python version here depending on if you  
         have it installed on your computer.
  
 Now, go to the train.py within PyCharm.  Here is where we train our data.  If you want to use our data that is fine!  It was all open source found at: 
   https://www.gutenberg.org/
   
 If you don't want to use the same data go to "storyData.txt" delete what we currently have and replace with the data that you want. 
 
 The lines that you can edit within "train.py" are 51, 52, and 65.
 
   Line 27 --> the sequence length
  
   Line 51 --> relates to the size of the rnn
  
   Line 52 --> should have the same rnn size
  
   Line 65 --> you can change the batch size and/or the epochs size
   
When everything is changed, you can run some tests.  Please note, that with the current settings it took my computer 4 hours to run one epochs- this code takes a lot of time.  However you can lower some of the numbers (for example rnn) to speed up the process.  Just note that some of the data you will get from those tests will not be as accurate.

While this is running it will be saving files, its a good idea to keep track of which one holds both the best loss and accurancy while the code runs and pick that one for when we move onto the write file.
