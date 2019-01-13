# Get datasets
import os
import tarfile
import pandas as pd
from six.moves import urllib
import matplotlib.pyplot as plt
import numpy as np

from my_preprocessor import CombinedAttributesAdder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import CategoricalEncoder
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets","housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

#Download tgz file named "housing.tgz" from url to path
def fetch_housing_data (housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

#fetch_housing_data() #get the files

#load csv file
def load_csv_file(path = HOUSING_PATH, file_name = "housing.csv"):
    csv_path = os.path.join(path, file_name)
    return pd.read_csv(csv_path)

housing_df = load_csv_file()

def print_housing_hist():
    housing_df.hist(bins=50, figsize=(20,15))
    plt.show()
#print_housing_hist()
#print(load_csv_file().head())
#print(housing_df.info())
#print(housing_df["ocean_proximity"].value_counts())
#print(housing_df.describe())

#Make testsets from whole data
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data)) #data shuffle
    test_set_size = int(len(data) * test_ratio) # whole_data * test_ratio
    test_indices = shuffled_indices[:test_set_size] # index from 0 ~ test_set_size
    train_indices = shuffled_indices[test_set_size:] #index from test_set_size ~ end of data
    return data.iloc[train_indices], data.iloc[test_indices]
#train_set , test_set = split_train_test(housing_df, 0.2)

#print(len(train_set), "train +", len(test_set) , "test")
#It has a problem : If a column's data ratio is 3:7, random data split can make this dis

#stratified sampling
#make stratum
housing_df["income_cat"] = np.ceil(housing_df["median_income"] / 1.5)
housing_df["income_cat"].where(housing_df["income_cat"] < 5, 5.0, inplace =True)

#housing_df["income_cat"].hist() #make category per 1.5. max is 5.0

housing_df = housing_df.reset_index()
split = StratifiedShuffleSplit(n_splits=1, test_size= 0.2, random_state= 42)

for train_index, test_index in split.split(housing_df,housing_df["income_cat"]):
    start_train_set = housing_df.loc[train_index]
    start_test_set = housing_df.loc[test_index]

#print(housing_df["income_cat"].value_counts() / len(housing_df))
#print(start_train_set["income_cat"].value_counts() / len(start_train_set))
#start_train_set.plot(kind ="scatter", x = "longitude", y ="latitude", alpha =0.1, s = start_train_set["population"] / 100, label = "population", figsize = (10,7), c = "median_house_value", cmap = plt.get_cmap("jet"), colorbar=True, sharex = False)
#housing_df.plot(kind ="scatter", x = "longitude", y ="latitude")

#corr_matrix = start_train_set.corr()
pd.set_option("display.max_rows", 15)
pd.set_option("display.max_columns",15)

#print(corr_matrix["median_house_value"].sort_values(ascending=False))

#Make train set from data : delete target column
train_set = start_train_set.drop("median_house_value", axis = 1)
train_label = start_train_set["median_house_value"].copy()

#Data preprocessing : handle non-int values

#in case of nan
#choices 1. drop column, 2. dropna, 3. fillna

#in case of non-int values

print(train_set["ocean_proximity"].head(10))
# Items to index

##Step 1.
house_cat_encoded, house_categories = train_set["ocean_proximity"].factorize()
np.set_printoptions(threshold = 15)
print(house_cat_encoded[:10])
print(house_categories)
##Step 2.
encoder = OneHotEncoder(categories = "auto")
house_cat_1hot = encoder.fit_transform(house_cat_encoded.reshape(-1,1))
print(house_cat_1hot.toarray())
#----------------------------------------------------------------------------------

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(train_set.values)
print(housing_extra_attribs)