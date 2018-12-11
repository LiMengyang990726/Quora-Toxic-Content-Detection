# python standard library
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
# if any of the above modules is not found, can simply do  `pip install`

# ==============================================================================
# Common packages introduction in Data Science and Machine Learning
# ==============================================================================
# pandas: operating with dataframe and data analysis
# numpy: high-level mathematical operations on metrics and arrays
# matplotlib: 2D plotting
# seaborn: based on matplotlib, higher level plotting
# sklearn: a machine learning library
# keras: a deep learning library with self-customizable layer
# tensorflow has the similar features with sklearn and keras
# ==============================================================================

sklearn_housing_bunch = fetch_california_housing("~/data/sklearn_datasets/")
print(sklearn_housing_bunch.DESCR)
print(sklearn_housing_bunch.feature_names)
sklearn_housing = pandas.DataFrame(sklearn_housing_bunch.data,columns=sklearn_housing_bunch.feature_names)
# .info() show the data type
sklearn_housing.info()
# .describe() show the statistics
sklearn_housing.describe()
sklearn_housing.head(5)
sklearn_housing['AveRooms'].value_counts()

sklearn_housing.hist(bins=50,figsize=(20,15))
plt.show()

# To prevent data snooping bias, we will randomly select 20% of the dataset as the test
# set, and never touch the test set
import numpy as np
def split_train_test(data,test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

# while the above function will generate the different dataset every time run
# the program, and the solutions will break down when we have a new dataset
# the common solution is shown as below
import hashlib
def test_set_check(identifier,test_ratio,hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio
def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_:test_set_check(id_,test_ratio,hash))
    return data.loc[~in_test_set],data.loc[in_test_set]

# the index as the identifier housing_with_id = housing.reset_index()
# (simpliest, while when appending new rows, there is a possiblity that previous rows got deleted)
# a more stable way of setting identifier
housing_with_id = sklearn_housing.reset_index()
housing_with_id["id"] = sklearn_housing["Longitude"] * 1000 + sklearn_housing["Latitude"]
train_set,test_set = split_train_test_by_id(housing_with_id,0.2,"id")
len(train_set)
len(test_set)

# alternatively, can directly use the scikit-learn method
# from sklearn.model_selection import train_test_split
# train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)

# to extract data based on stratified sampling (from different catagory based on certain percentage)
# cross validation: divide the training set into k folds, train on all permuatation
# of leaving one fold as test set, then average the error
sklearn_housing["income_cat"] = np.ceil(sklearn_housing["MedInc"] / 1.5)
sklearn_housing["income_cat"].where(sklearn_housing["income_cat"] < 5, 5.0, inplace = True)
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index, test_index in split.split(sklearn_housing, sklearn_housing["income_cat"]):
    strat_train_set = sklearn_housing.loc[train_index]
    strat_test_set = sklearn_housing.loc[test_index]
sklearn_housing["income_cat"].value_counts() / len(sklearn_housing)
# to drop the income_cat column
# for set in (strat_train_set, strat_test_set):
#   set.drop(["income"], axis = 1, inplace = True)
housing = strat_train_set.copy()
housing.plot(kind = "scatter", x = "Longitude", y = "Latitude", alpha = 0.1,
            s = housing["Population"]/100, label = "Population",
            c = "median_house_value", cmap = plt.get_cmap("jet"),
            colorbar = True)
plt.show()
housing.head()
