# -*- coding: utf-8 -*-
"""InsuranceCostPrediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1dSx1Iac_4xueBcxi4jKGzShelI2U-vra

### Importing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from google.colab import drive
drive.mount('/content/drive')

# To get this dataset 
# https://drive.google.com/file/d/1g179h0nV7wRCNSFWLMXlraIBVfA6aeBY/view?usp=sharing

# Data collection and processing
insurance_dataset=pd.read_csv("/content/drive/MyDrive/ML Projects/insurance.csv")
insurance_dataset.head()

# no. of rows and columns
insurance_dataset.shape

# Getting some info. about dataset
insurance_dataset.info()

"""categorical features:
- Sex/Gender
- Smoker
- Region

Target: Charges
"""

# Checking for missing value
insurance_dataset.isnull().sum()

"""### Data Analysis"""

# Statistical measure of the dataset
insurance_dataset.describe()

# Distrubution of age value
sns.set()
plt.figure(figsize=(8,6))
sns.distplot(insurance_dataset['age'])

plt.title('Age distribution')
plt.show()

# Gender column
plt.figure(figsize=(6,6))
sns.countplot(x='sex',data=insurance_dataset,palette="viridis")

plt.title('Gender distribution')
plt.show()

insurance_dataset['sex'].value_counts()

# BMI distribution in dataset
plt.figure(figsize=(8,6))
sns.distplot(insurance_dataset['bmi'])

plt.title('BMI distribution')
plt.show()

"""Normal BMI range : 18.5 -> 25"""

# Children column
plt.figure(figsize=(6,6))
sns.countplot(x='children',data=insurance_dataset,palette="viridis")

plt.title('children')
plt.show()

insurance_dataset['children'].value_counts()

# Smoker column
plt.figure(figsize=(6,6))
sns.countplot(x='smoker',data=insurance_dataset,palette="viridis")

plt.title('Smoker distribution')
plt.show()

insurance_dataset['smoker'].value_counts()

# Region column
plt.figure(figsize=(6,6))
sns.countplot(x='region',data=insurance_dataset,palette="viridis")

plt.title('Region distribution')
plt.show()

insurance_dataset['region'].value_counts()

# For age and BMI count plot will not be good because of many values
# USe distribution plot

# Charges distribution
plt.figure(figsize=(8,6))
sns.distplot(insurance_dataset['charges'])

plt.title('Charge distribution')
plt.show()

"""## Data Pre-processing

### Encoding the categorical features:
"""

# Encoding the gender/sex column
insurance_dataset.replace({'sex':{'male':0,'female':1}},inplace=True)

# Encoding smoker column
insurance_dataset.replace({'smoker':{'yes':0,'no':1}},inplace=True)

# Encoding region column
insurance_dataset.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}},inplace=True)

insurance_dataset

# Splitting the features and target
X=insurance_dataset.drop(columns='charges',axis=1)  #X contains all the other features not target
Y=insurance_dataset['charges']  #Y contains the targert(charges )
print(X)

print(Y)

# Splitting the data into training data and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,  random_state=2)

# Knowing the shape of our train and test data
print(X.shape,X_train.shape, X_test.shape)
print(Y.shape,Y_train.shape, Y_test.shape)

"""## Model Training

"""

# Linear regression model training

# Loading the Linear regression model
regressor= LinearRegression()

# Fitting the training data into regress or to make the line of regression by using the points
regressor.fit(X_train,Y_train)

# Now model has been trained and its time to evaluate/test

"""### Model Evaluation"""

# Prediction on training data
training_data_prediction = regressor.predict(X_train)

# R squared value : if value is close to 1 then our model is performing well.
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R Squared value : ',r2_train)

# Prediction on testing  data
test_data_prediction = regressor.predict(X_test )

# R squared value : if value is close to 1 then our model is performing well.
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R Squared value : ',r2_test)

"""## Building a prediction system"""

# for input
# 'male':0,'female':1
# 'yes':0,'no':1
# 'southeast':0,'southwest':1,'northeast':2,'northwest':3
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Building a system which can predict insurance based on the features in input
input_data=(25,0,26.22,0,1,2)

# changing input_data (tuple) to numpy array
input_data_np=np.asarray(input_data)

input_data_np

# Reshaping the array
input_data_reshaped = input_data_np.reshape(1,-1)
input_data_reshaped

# Predicting from the model
prediction = regressor.predict(input_data_reshaped)
print('The insurance cost is : ',prediction[0])