from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

#Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='constant')

#Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[
	('imputer', SimpleImputer(strategy='most_frequent'),
	('onehot', OneHotEncoder(handle_unknown='ignore'))
]

#Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(
	transformers = [
		('num', numerical_transformer, numerical_cols),
		('cat', categorical_transformer, categorical_cols)
	]
)

from sklearn.ensemble import RandomForest Regressor()

model = RandomForestRegressor(n_estimators=100, random_state= 0


from sklearn.metrics import mean_absolute_error

#Bunlde preprocessing and modeling code in a pipeline

my_pipeline = Pipeline(steps = [('preprocessor', preprocessor),
				('model', model)
				])

# Preprocessing of training data, fit model
my_pipeline.fit(X_train, y_train)

#Preprocessing of validation data, get predctions
preds = my_pipeline.predict(X_valid, y_train)

# Evaluate the model

score = mean_absolute_error(y_valid, preds)

print('MAE: ', score)
