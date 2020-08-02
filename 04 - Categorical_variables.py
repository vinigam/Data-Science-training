#Categorical variables

# Drop these columns

# Label encoding
	# - Create an column with crescente numbers that represent the respective categorical
	# - with an order sense ('Never' is less than 'Frequently')
# One hot encoding 
	# - Create columns for each variable and mark with 1 the order they appears
	# - no order sense ('Yellow' is neither more nor less than 'Red')
	# - Dont work well with categorical variables that can handle a large number of values, 15+

#----------------------------------------------------

#Get list of categorical variables

s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables: ")
print(object_cols)
#-----------------------------------------------------
#Approaches
#-----------------------------------------------------
#drop
drop_X_train = X_train.select_dtypes(exclude['object'])
drop_X_val = X_train.select_dtypes(exclude['object'])

print(score_dataset(drop_X_train, drop_X_val, y_train, y_val)

MAE from Approach 1 (Drop categorical variables):
175703.48185157913

#-----------------------------------------------------

# Label encoding

from sklearn.preprocessing import LabelEncoder

# Make copy to avoid changin original data

label_X_train = X_train.copy()
label_X_val = X_val.copy()

# Apply label encoder to each column with categorical ddata

label_encoder = LabelEncoder()

for col in object_cols:
	label_X_train[col] = label_encoder.fit_transform(X_train[col])
	label_X_val[col] = label_encoder.transform(X_valid[col])

print("MAE from Approach 2 (Label Encoding):") 
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))
MAE from Approach 2 (Label Encoding):
165936.40548390493


#-----------------------------------------------
# One hot encoder

from sklearn.preprocessing import OneHotEncoder

# We set handle_unknown='ignore' to avoid errors when the validation data contains classes that aren't represented in the training data, and
# setting sparse=False ensures that the encoded columns are returned as a numpy array (instead of a sparse matrix).

OH_encoder = OneHotEncoder(handle_unknown ='ignore, sparse =False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_col])
OH_cols_val = pd.DataFrame(OH_encoder.transform(X_val[object_col])

# One-hot encoding removed index; put it back
	
OH_cols_train.index = X_train.index
OH_cols_val.index = X_val.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_val = X_val.drop(object_cols, axis =1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis = 1)
OH_X_valid = pd.concat(num_X_val, OH_cols_val], axis = 1)

print("MAE from Approach 3 (One-Hot Encoding):") 
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))
MAE from Approach 3 (One-Hot Encoding):
166089.4893009678