import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing
from tqdm import tqdm # show the processing bar
import math
from sklearn.model_selection import train_test_split

# keras: High level deep learning API, build on top of Tensorflow, CNTK, or Theano
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

# create a test set for 
