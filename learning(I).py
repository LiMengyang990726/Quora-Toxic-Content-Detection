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
#
# Common packages introduction in Data Science and Machine Learning
#
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
#
# Knowing Your Data
#
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
housing['households'].value_counts()

housing.hist(bins=50,figsize=(20,15))
plt.show()

# ==============================================================================
#
# Creating a test set, and Set Aside
#
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
#
# Stratified Sampling
#
# ==============================================================================
# To get random data from all catagories
# ==============================================================================
# to extract data based on stratified sampling (from different catagory based on certain percentage)
<<<<<<< HEAD
=======
# cross validation: divide the training set into k folds, train on all permuatation
# of leaving one fold as test set, then average the error
>>>>>>> 25c40f26adaa6817c8d4f872646640feff7cf764
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
#
# Observe Correlation
#
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
#
# Data Cleaning
#
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

# ==============================================================================
# Step 2: Transformer
# ==============================================================================
# transform text into categorical attributes
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

# or, can self-define transformer with BaseEstimator
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3,4,5,6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # by default, true
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        rooms_per_household=X[:,rooms_ix]/X[:,household_ix]
        population_per_household=X[:,population_ix]/X[:,household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room=X[:,bedrooms_ix]/X[:,rooms_ix]
            return np.c_[X,rooms_per_household,population_per_household,bedrooms_per_room]
        else:
            return np.c_[X,rooms_per_household,population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs=attr_adder.transform(housing.values)
housing_extra_attribs

# Together with transforming, scaling issue will be addressed here as well
# First method: Min-max scaling / Normalization
#               (Number - MinValue) / (MaxValue - MinValue)
#               Scikit-learn: MinMaxScaler
# Second method: Standardization [Provide zero mean and unit variance]
#                (Number - mean) / variance
#                Scikit-learn: StandardScaler
# Be sure to fit scaler to training data only

# ==============================================================================
# The whole transforming and scaling process for numerical value
# can be done using Pipeline class provided by scikit-learn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline=Pipeline([
            ('imputer',Imputer(strategy="median")),
            ('attr_adder',CombinedAttributesAdder()),
            ('std_scaler',StandardScaler()),
            ])
housing_num_tr=num_pipeline.fit_transform(housing_num)
# this fit will call all fit_transform method sequentially

# ==============================================================================
# Whole transforming and scaling process for both numerical value and text
# can be done using FeatureUnion class provided by scikit-learn
# all transformers will be run in parallel and results will be cancatenated finally
from sklearn.pipeline import FeatureUnion
<<<<<<< HEAD
from sklearn_features.transformers import DataFrameSelector
=======
>>>>>>> 25c40f26adaa6817c8d4f872646640feff7cf764

num_attribs=list(housing_num)
cat_attribs=["ocean_proximity"]

<<<<<<< HEAD
# for solving the issue brought by the misuse of LabelBinarizer
class CustomLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, sparse_output=False):
        self.sparse_output = sparse_output
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        enc = LabelBinarizer(sparse_output=self.sparse_output)
        return enc.fit_transform(X)

num_pipeline=Pipeline([
            ('selector',DataFrameSelector(num_attribs)),
=======
num_pipeline=Pipeline([
            ('selector',num_attribs),
>>>>>>> 25c40f26adaa6817c8d4f872646640feff7cf764
            ('imputer',Imputer(strategy="median")),
            ('attr_adder',CombinedAttributesAdder()),
            ('std_scaler',StandardScaler()),
            ])
cat_pipeline=Pipeline([
            ('selector',DataFrameSelector(cat_attribs)),
            ('label_binarizer',CustomLabelBinarizer()),
            ])
full_pipeline=FeatureUnion(transformer_list=[
            ("num_pipeline",num_pipeline),
            ("cat_pipeline",cat_pipeline),
            ])

housing_prepared=full_pipeline.fit_transform(housing)
housing_prepared

# ==============================================================================
#
# Fit ML models
#
# ==============================================================================

# try linear regression model
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)
some_data=housing.iloc[:5]
some_labels=housing_labels.iloc[:5]
some_data_prepared=full_pipeline.fit_transform(some_data)
some_data_prepared.shape
print(lin_reg.predict(some_data_prepared))

from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
housing_labels
lin_mse=mean_squared_error(housing_labels,housing_predictions)
lin_rmse=np.sqrt(lin_mse)
lin_rmse

# try decision tree model
from sklearn.tree import DecisionTreeRegressor
tree_reg=DecisionTreeRegressor()
tree_reg.fit(housing_prepared,housing_labels)
tree_predictions=tree_reg.predict(housing_prepared)
tree_mse=mean_squared_error(housing_labels,tree_predictions)
print(np.sqrt(tree_mse))

# to get better evaluation result, use cross-validation
# cross validation: divide the training set into k folds, train on all permuatation
# of leaving one fold as test set, then average the error
from sklearn.model_selection import cross_val_score
scores=cross_val_score(tree_reg,housing_prepared,housing_labels,
                       scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(scores)
rmse_scores
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:",scores.mean())
    print("Standard deviation:",scores.std())
display_scores(tree_rmse_scores)
