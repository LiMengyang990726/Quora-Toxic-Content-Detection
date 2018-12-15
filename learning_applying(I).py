import os
import tarfile # for operating on tar files
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
from http import HTTPStatus

# from pypi
import matplotlib
import pandas
import requests
import seaborn
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from tabulate import tabulate

qcomment = pandas.read_csv("./train.csv")
qcomment.describe()
qcomment.info()

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(qcomment, test_size = 0.2, random_state = 42)
train_set.info()
len(train_set)

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
