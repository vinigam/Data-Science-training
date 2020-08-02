# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

data = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        data.append(pd.read_csv(os.path.join(dirname, filename)))
    
#get data

gender = data[0]
test = data[1]
train = data[2]

#Binary sex
def binary_sex(val):
    if val == 'female':
        return 1
    else:
        return 0

train['BinarySex'] = train['Sex'].map(binary_sex)
test['BinarySex'] = test['Sex'].map(binary_sex)
            

#features
features = ['Pclass','Parch','Age', 'SibSp', 'BinarySex']

filttered_train = train[features]
filttered_test = test[features]

# missing values
missing_train = filttered_train.isnull().sum()
missing_test = filttered_test.isnull().sum()

#dealing with missing values with imputer method

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

train_plus = filttered_train.copy()
test_plus = filttered_test.copy()

inputed_train = pd.DataFrame(my_imputer.fit_transform(train_plus))
inputed_test = pd.DataFrame(my_imputer.transform(test_plus))

new_missing_train = inputed_train.isnull().sum()
new_missing_test = inputed_test.isnull().sum()


inputed_train.columns = filttered_train.columns
inputed_test.columns = filttered_test.columns

#Test predicts with train test split
# X = inputed_train, y = train.Survived
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(inputed_train, train.Survived, random_state = 1)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
import numpy as np
depth = [50, 100, 500, 1000]

#decide the best depth

for max_depth in depth:
    model = RandomForestClassifier(max_leaf_nodes = max_depth, random_state = 0)
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    print(np.mean(y_val == pred))


#final model

final_model = RandomForestClassifier(max_leaf_nodes = 50, random_state = 0)
model.fit(inputed_train, train.Survived)
pred = model.predict(inputed_test)

#submission
sub = pd.Series(pred, index=test['PassengerId'], name='Survived')
sub.shape

sub.to_csv("submission.csv", index=True)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session