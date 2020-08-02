# Code you have previously used to load data
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)
# Create target object and call it y
y = home_data.SalePrice
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify Model
iowa_model = DecisionTreeRegressor(random_state=1)
# Fit Model
iowa_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE: {:,.0f}".format(val_mae))

# for loop in this function to get the best mae
def get_mae(max_leaf_nodes, train_X, val_x, train_y, valy):
    model = DecisionTreeRegressor(max_leaft_node = max_leaf_nodes, random_state = 0)
    model.fit(train_X, train_y)
    val_predict = model.predict(val_X)
    mae = mean_absolute_error(val_y, val_predict)
    return mae

# best mae
    candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
value = 0
index = 0
for max_leaf_size in candidate_max_leaf_nodes:
    temp = get_mae(max_leaf_size, train_X, val_X, train_y, val_y)
    print(temp)
    if value == 0:
        value = temp
        index = max_leaf_size
    elif temp < value:
        value = temp
        index = max_leaf_size
    
# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
best_tree_size = index
# Best model accuracy
final_model = DecisionTreeRegressor(best_tree_size,random_state = 0)
final_model.fit(X,y)

