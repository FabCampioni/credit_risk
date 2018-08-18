# Kaggle Competition Entry
# Home Credit Default Risk
# Predict how capable each applicant is of repaying a loan

# Fabio Campioni
# August 18, 2018

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier

# Load training data
train_data = pd.read_csv('application_train.csv')

# Load test data set
test_data = pd.read_csv('application_test.csv')

# Set index
train_data.set_index('SK_ID_CURR', inplace=True)
test_data.set_index('SK_ID_CURR', inplace=True)

# Keywords which identify numerical variables
keywords = ["AMT","AVG","MEDI","MODE", "CNT", "DAYS", "AGE", "EXT"]

numeric_variables = [column for column in train_data.keys() if any(k in column for k in keywords)]

# Anything non numeric is categorical
categorical_variables = [column for column in train_data.keys() if column not in numeric_variables]

# Some of the MODE variables are actually categorical so manually reassign them
numeric_variables.remove("FONDKAPREMONT_MODE")
categorical_variables.append("FONDKAPREMONT_MODE")

numeric_variables.remove("HOUSETYPE_MODE")
categorical_variables.append("HOUSETYPE_MODE")

numeric_variables.remove("WALLSMATERIAL_MODE")
categorical_variables.append("WALLSMATERIAL_MODE")

numeric_variables.remove("EMERGENCYSTATE_MODE")
categorical_variables.append("EMERGENCYSTATE_MODE")

# Fill NAs and convert each column depending on its type
for column in train_data:
    if column in numeric_variables:
        train_data[column].fillna('0', inplace=True)
        train_data[column].astype('int64', inplace=True)
    else:
        train_data[column].fillna('', inplace=True)
        le = LabelEncoder()
        train_data[column] = le.fit_transform(train_data[column])

for column in test_data:
    if column in numeric_variables:
        test_data[column].fillna('0', inplace=True)
        test_data[column].astype('int64', inplace=True)
    else:
        test_data[column].fillna('', inplace=True)
        le = LabelEncoder()
        test_data[column] = le.fit_transform(test_data[column])


# Drop the target variable from the training data
X = train_data.drop(['TARGET'], axis=1)

# Isolate the target variable
Y = train_data.TARGET

# Split the training dataset into an 80/20 split to get an idea of prediction accuracy
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Train model with Gradient Boosting Classifier (ensemble based learner)
gbr = GradientBoostingClassifier()

# Fit on training data
gbr.fit(X_train, y_train)

# Score model on held back train data
train_score = gbr.score(X_test, y_test)

# Make predictions using test data
test_data['TARGET'] = gbr.predict_proba(test_data)[:,1]

header = ['TARGET']
test_data.to_csv('result.csv', columns=header)