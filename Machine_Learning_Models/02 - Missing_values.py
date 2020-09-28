#Missing values
#-- drop columns --
# drop columns with missing value

cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any())] #any return true if all axis (column) is true

reduced_X_train = X_train.drop(cols_with_missing, axis = 1)
reduced_X_val = X_val.drop(cols_with_missing, axis = 1)

#--imput--
#fill the missing values with values or/and create new columns with true or false 'was_missing'

from sklearn.impute import SimpleImputer

X_train_plus = X_train.copy()
X_val_plus = X_val.copy()

myimputer = SimpleImputer()SD

for col in cols_with_missing:
	X_train_plus[cols +['_was_missing'] = X_train_plus[cols].isnull()
	X_val_plus[cols +'_was_missing'] = X_val_plus[cols].isnull()

imputed_X_train_plus = pd.DataFrame(myimputer.fit_transform(X_train_plus))
imputed_X_val_plus = pd.DataFrame(myimputer.transform(X_val_plus))

imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_val_plus.columns = X_val_plus.columns

score_dataset(imputed_X_train_plus, imputed_X_val_plus, y_train_plus, y_val_plus)
