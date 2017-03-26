# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 14:18:48 2017

@author: Arthur
"""

# impot lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data_set = pd.read_csv('Salary_Data.csv')
X = data_set.iloc[:, :-1].values
Y = data_set.iloc[:, -1].values


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1.0/3, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Exp')
plt.xlabel('Exp')
plt.ylabel('salary')
plt.show()