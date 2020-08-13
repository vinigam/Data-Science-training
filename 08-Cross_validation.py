#Cross validation
#Used to small datasets

#Divide the datasets in 5 groups, using each one as validation and the rest as training


from skelearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
form sklearn.impute import SimpleImputer

my_pipeline = Pipeline(steps=[
		('preprocessor', SimpleImputer()),
		('model', RandomForestRegressor(n_estimators=50, random_state = 0)
		
	])

from sklearn.model_selection import cross_val_score

scores = -1 * cross_val_score(my_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')

print(scores)
print('Average MAE : {}'.format(scores.mean()))