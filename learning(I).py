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
class Data:
    source_slug = "../data/california-housing-prices/"
    target_slug = "../data_temp/california-housing-prices/"
    url = "https://github.com/ageron/handson-ml/raw/master/datasets/housing/housing.tgz"
    source = source_slug + "housing.tgz"
    target = target_slug + "housing.csv"
    chunk_size = 128
def get_data():
    """Gets the data from github and uncompresses it"""
    if os.path.exists(Data.target):
	    return

    os.makedirs(Data.target_slug, exist_ok=True)
    os.makedirs(Data.source_slug, exist_ok=True)
    response = requests.get(Data.url, stream=True)
    assert response.status_code == HTTPStatus.OK
    with open(Data.source, "wb") as writer:
	    for chunk in response.iter_content(chunk_size=Data.chunk_size):
	        writer.write(chunk)
    assert os.path.exists(Data.source)
    compressed = tarfile.open(Data.source)
    compressed.extractall(Data.target_slug)
    compressed.close()
    assert os.path.exists(Data.target)
    return

sklearn_housing_bunch = fetch_california_housing("~/data/sklearn_datasets/")
print(sklearn_housing_bunch.DESCR)
print(sklearn_housing_bunch.feature_names)
sklearn_housing = pandas.DataFrame(sklearn_housing_bunch.data,columns=sklearn_housing_bunch.feature_names)
get_data()
housing = pandas.read_csv(Data.target)

# ==============================================================================
# Knowing Your Data
# ==============================================================================
# .info() show the data type
# .describe() show the statistics
# .head() simply show
# .value_counts
# simple plotting will also be done to know data
# ==============================================================================
housing.info()
housing.describe()
housing.head(5)
housing['AveRooms'].value_counts()

housing.hist(bins=50,figsize=(20,15))
plt.show()

# ==============================================================================
# Creating a test set, and Set Aside
# ==============================================================================
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
housing_with_id = housing.reset_index()
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set,test_set = split_train_test_by_id(housing_with_id,0.2,"id")
len(train_set)
len(test_set)

# alternatively, can directly use the scikit-learn method
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)

# ==============================================================================
# Stratified Sampling
# ==============================================================================
# To get random data from all catagories
# ==============================================================================
# to extract data based on stratified sampling (from different catagory based on certain percentage)
# cross validation: divide the training set into k folds, train on all permuatation
# of leaving one fold as test set, then average the error
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace = True)
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
housing["income_cat"].value_counts() / len(housing)

# to drop the income_cat column
for set in (strat_train_set, strat_test_set):
  set.drop(["income_cat"], axis = 1, inplace = True)

housing.plot(kind = "scatter", x = "longitude", y = "latitude", alpha = 0.5,
            s = housing["population"]/100, label = "Population",
            c = "median_house_value", cmap = plt.get_cmap("jet"),
            colorbar = True)
plt.show()

# ==============================================================================
# Observe Correlation
# ==============================================================================
# .corr() get the correlation matrix across all columns
# create some more meaningful columns based on currect columns
# .corr() to observe the correlation see if the newly added columns have a
# stronger correlation
# -1: strongly negative correlation ; 1: strongly positive correlation
# ==============================================================================
# Correlation: only measures linear correlation
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=True)

from pandas.tools.plotting import scatter_matrix
attributes = ["median_house_value","median_income",
                "total_rooms","housing_median_age"]
scatter_matrix(housing[attributes],figsize = (12,8))
plt.show()
# zoom in the most promising attribute: median income
housing.plot(kind = "scatter", x = "median_income", y = "median_house_value", alpha = 0.1, figsize = (12,8))
plt.show()

housing.info()
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=True)

housing = strat_train_set.drop("median_house_value",axis = 1)
housing_labels = strat_train_set["median_house_value"].copy()

# ==============================================================================
# Data Cleaning
# ==============================================================================
# Step 1: deal with missing values
# scikit learn provides: Imputer instance to specially handle missing values
from sklearn.preprocessing import Imputer
housing_num = housing.drop("ocean_proximity", axis=1)
imputer = Imputer(strategy="median")
imputer.fit(housing_num)
imputer.statistics_
housing_num.median().values
X =imputer.transform(housing_num)
housing_tr = pandas.DataFrame(X, columns=housing_num.columns)

# Step 2: transform text into categorical attributes
# scikit-learn provide: LabelEncoder
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)
housing_cat_encoded
encoder.classes_
# to provide more meaningful numerical representation to text
# use "one-hot encoding"
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
housing_cat_1hot
# to finish the two steps transformation, we can use LabelBinarizer
# two steps: text to number, number to meaningful number/one-hot vector
# LaebelBinarizer will return a dense NumPy array
# while a sparse matrix is more storage efficient
# to transform to sparse matrix: parsing "sparse_output=True"
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer(sparse_output=True)
housing_cat_1hot = encoder.fit_transform(housing_cat)
housing_cat_1hot
