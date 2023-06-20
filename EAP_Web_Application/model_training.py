import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle



# reading the data
train_data = pd.read_csv('Emp_attrition_Train.csv')

#dataset overview
train_data.info()

train_data.drop(['Ename','EmployeeNumber'],axis=1,inplace=True)

#statistical view of dataset
train_data.describe()

"""EXPLORATORY DATA ANALYSIS"""

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numeric_df = train_data.select_dtypes(include=numerics)
numeric_df.columns

category_df = train_data.select_dtypes(include='object')
category_df

#finding the correlation between the columns
corr=numeric_df.corr()

sns.heatmap(corr)
plt.show()

"""There are high correlation between the columns

## Feature engineering and categorical encoding-Training
"""

# Define a dictionary for the target mapping
target_map = {'Yes':1, 'No':0}
# Use the pandas apply method to numerically encode our attrition target variable
train_data["Attrition_numerical"] = train_data["Attrition"].apply(lambda x: target_map[x])

# Drop the Attrition_numerical column from attrition dataset first - Don't want to include that
train_attrition = train_data.drop(['Attrition_numerical'], axis=1)


### Training data....
# Empty list to store columns with categorical data
categorical = []
for col, value in train_attrition.items():
    if value.dtype == 'object':
        categorical.append(col)

# Store the numerical columns in a list numerical
numerical = train_attrition.columns.difference(categorical)

# Store the categorical data in a dataframe called attrition_cat
train_attrition_cat = train_attrition[categorical]
train_attrition_cat = train_attrition_cat.drop(['Attrition'], axis=1) # Dropping the target column



train_attrition_cat = pd.get_dummies(train_attrition_cat)
train_attrition_cat.head(3)



# Store the numerical features to a dataframe attrition_num
train_attrition_num = train_attrition[numerical]

# Concat the two dataframes together columnwise
train_attrition_final = pd.concat([train_attrition_num, train_attrition_cat], axis=1)

#Define a dictionary for the target mapping
target_map = {'Yes':1, 'No':0}
# Use the pandas apply method to numerically encode our attrition target variable
train_target = train_attrition["Attrition"].apply(lambda x: target_map[x])
train_target.head(3)

"""## Naive Bayes classifier"""

from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()

nb.fit(train_attrition_final.values, train_target)
print('Model training completed............')


print('saving the model...')

pickle.dump(nb, open('model.pkl', 'wb'))

