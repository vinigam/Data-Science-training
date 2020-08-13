#XGBoost (Gradient boosting)

# model 
#	make predictions
#		add the predictions to ensemble and calculate loss
#			use the result to fit a new model
#				add new model to ensemble
#					repeat 



#(n_estimators), define how many cicles of model iterativity will happens
#(early_stopping_rounds), automatically, find a ideal n_estimators value because stops when the score deteriorate X times estimated
#(learning_rate), way to get the score multiplying an little value per each tree added, this could avoid overfitting with early_stopping_rounds and improve the model
small learning rate will be more accurate but will be more longer to train(as default = learning_rate= 0.1)
#(n_jobs) way to optimize the fitting in large datasets

from xgboost import XGBRegressor

my_model = XGBRegressor(n_estimators=500, learning_rate= 0.05, n_jobs = 4)
my_model.fit(X_train, y_train,
	     early_stopping_rounds=5,
	     eval_set=[(X_valid, y_valid)],
	     verbose = False
		)

from sklearn.metrics import mean_absolute_error

pred = my_model.predict(X_valid)

print(mean_absolute_error(y_valid, pred))