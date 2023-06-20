
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

""" Model loading..."""

with open('model.pkl','rb') as f:
    nb = pickle.load(f)
"""Test data"""
target_map = {'Yes':1, 'No':0}
test_data =pd.read_csv('Emp_attrition_Test.csv')
test_data.drop(['Ename','EmployeeNumber'],axis=1,inplace=True)

test_data["Attrition_numerical"] = test_data["Attrition"].apply(lambda x: target_map[x])
test_attrition = test_data.drop(['Attrition_numerical'], axis=1)

#Testing data.....
# Empty list to store columns with categorical data
categorical = []
for col, value in test_attrition.items():
    if value.dtype == 'object':
        categorical.append(col)

# Store the numerical columns in a list numerical
numerical = test_attrition.columns.difference(categorical)

# Store the categorical data in a dataframe called attrition_cat
test_attrition_cat = test_attrition[categorical]
test_attrition_cat = test_attrition_cat.drop(['Attrition'], axis=1) # Dropping the target column

test_attrition_cat = pd.get_dummies(test_attrition_cat)
test_attrition_cat.head(10)


# Store the numerical features to a dataframe attrition_num
test_attrition_num = test_attrition[numerical]

# Concat the two dataframes together columnwise
test_attrition_final = pd.concat([test_attrition_num, test_attrition_cat], axis=1)

#Define a dictionary for the target mapping
target_map = {'Yes':1, 'No':0}
# Use the pandas apply method to numerically encode our attrition target variable
test_target = test_attrition["Attrition"].apply(lambda x: target_map[x])
test_target.head(3)

results = nb.predict_proba(test_attrition_final)


results_1= results[:,1]

np.set_printoptions(suppress=True)
len(results)
# results

print(np.argmax(results))

test_attrition_final.shape

predictions = nb.predict(test_attrition_final.values)
print(predictions.shape)
print(test_target.shape)

from sklearn.metrics import accuracy_score
model_accuracy = accuracy_score(test_target,nb.predict(test_attrition_final.values))
print(model_accuracy)

np.set_printoptions(precision = 2, suppress = True)
results_1 = results_1.tolist()

test_data =pd.read_csv('Emp_attrition_Test.csv')
test_data['results'] = results_1
# test_data

final_data = test_data.sort_values(by=['results'],ascending=False).head(5)[['Ename','EmployeeNumber']].reset_index(drop=True)

print(final_data)