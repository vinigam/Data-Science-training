import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

#get data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

#set , get y and drop y from train_data
equal_cols = set(train_data)-set(test_data)
#{'SalePrice'}

y = train_data.SalePrice

train_data = train_data.drop('SalePrice', axis = 1)

# get object cols and non object cols

object_cols = [col for col in train_data.columns if train_data[col].dtype == 'object']
non_object_cols = [col for col in train_data.columns if train_data[col].dtype != 'object']

#missing data
missing_values_non_object = train_data[non_object_cols].isnull().sum()
missing_values_object = train_data[object_cols].isnull().sum()

#input method to non categorical variables


from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

train_copy = train_data.copy()
test_copy = test_data.copy()

inputed_non_object_train = pd.DataFrame(my_imputer.fit_transform(train_copy[non_object_cols]))
inputed_non_object_test = pd.DataFrame(my_imputer.transform(test_copy[non_object_cols]))

inputed_non_object_train.index = train_copy.index
inputed_non_object_test.index = test_copy.index

# categorical variables

# drop categorical variables missing values greater than 1000
train_copy2 = train_copy[object_cols]
train_copy2 = train_copy2.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis = 1)
test_copy2 = test_copy[object_cols]
test_copy2 = test_copy2.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis = 1)

#cardinality

low_cardinality = [col for col in test_copy2.columns if test_copy2[col].nunique() < 10]
high_cardinality = [col for col in test_copy2.columns if test_copy2[col].nunique() > 10]

#for col in test_copy2.columns:
#   if test_copy2[col].nunique() > 10:
#       print('{}, {}'.format(col, test_copy2[col].nunique()))

#label encoder to low to high cardinality


from sklearn.preprocessing import LabelEncoder

LB_encoder = LabelEncoder()

train_label_encoder = train_copy2.drop(low_cardinality, axis = 1)
test_label_encoder = test_copy2.drop(low_cardinality, axis = 1)

#fill the mising value of categorical data hih cardinality filling with the most common value

test_label_encoder['Exterior1st']=test_label_encoder['Exterior1st'].fillna('VinylSd')
test_label_encoder['Exterior2nd']= test_label_encoder['Exterior2nd'].fillna('VinylSd')

for col in set(high_cardinality):
    train_label_encoder[col] = LB_encoder.fit_transform(train_label_encoder[col])
    test_label_encoder[col] = LB_encoder.transform(test_label_encoder[col])



#variables to concact after

    # inputed_non_object_train
    # inputed_non_object_test

    # train_label_encoder
    # test_label_encoder

#One hot encoder approach for low cardinality categorical variables

from sklearn.preprocessing import OneHotEncoder

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

#get only the low cardinality cols
train_OH_encoder = train_copy2.drop(high_cardinality, axis = 1)
test_OH_encoder = test_copy2.drop(high_cardinality, axis = 1)

#deal with missing values using the most frequent variable
missing_train = [col for col in train_OH_encoder.columns if train_OH_encoder[col].isnull().any()]
missing_test = [col for col in test_OH_encoder.columns if train_OH_encoder[col].isnull().any()]


train_OH_encoder[missing_train] = train_OH_encoder[missing_train].fillna('U')
test_OH_encoder[missing_test] = test_OH_encoder[missing_test].fillna('U')

#doing this approach again with this variable cause still have some missing
missing_test2 = [col for col in test_OH_encoder.columns if test_OH_encoder[col].isnull().any()]
test_OH_encoder[missing_test2] = test_OH_encoder[missing_test2].fillna('U')


OH_encoder_train = pd.DataFrame(OH_encoder.fit_transform(train_OH_encoder))
OH_encoder_test = pd.DataFrame(OH_encoder.transform(test_OH_encoder))

OH_encoder_train.index = train_OH_encoder.index
OH_encoder_test.index = test_OH_encoder.index

#variables to concact after

    # inputed_non_object_train
    # inputed_non_object_test

    # train_label_encoder
    # test_label_encoder

    #OH_encoder_train
    #OH_encoder_test

X_train_filttered = pd.concat([inputed_non_object_train, train_label_encoder, OH_encoder_train], axis = 1)
X_test_filttered = pd.concat([inputed_non_object_test, test_label_encoder, OH_encoder_test], axis = 1)

#create a model

from sklearn.ensemble import RandomForestRegressor




model = RandomForestRegressor(max_leaf_nodes= 1500, random_state = 0)
model.fit(X_train_filttered, y)
pred = model.predict(X_test_filttered)

sub = pd.Series(pred, index = test_data["Id"], name = "SalePrice")

sub.to_csv("Home_Prices.csv", index=True)

