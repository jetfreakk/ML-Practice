#Simple Linear Regression

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the Datasets
dataset = pd.read_csv('G:\ML Practice\Machine Learning A-Z Template Folder\Part 2 - Regression\Section 4 - Simple Linear Regression\P14-Simple-Linear-Regression\Simple_Linear_Regression\Salary_Data.csv')
X= dataset.iloc[:,:-1].values
y= dataset.iloc[:,1].values

#Splitting the dataset into the Training and Test set
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=1/3, random_state =0)

#Fitting Simple Linear Regression to Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)
y_pred2 = regressor.predict(X_train)
#Visualising the Training Set Results
plt.scatter(X_train,y_train,color = 'red')
plt.plot(X_train,y_pred2,color='blue')
plt.title('Salary vs Experience[Training Set]')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()