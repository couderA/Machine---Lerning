# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 14:18:48 2017

@author: Arthur
"""

# impot lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data_set = pd.read_csv('Data.csv')
X = data_set.iloc[:, :-1].values
Y = data_set.iloc[:, -1].values
                 
                 
from sklearn.preprocessing import Imputer
inputer = Imputer()
inputer.fit(X[:, 1:3])
X[:, 1:3] = inputer.transform(X[:, 1:3])


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_contry = LabelEncoder()
X[:, 0] = label_encoder_contry.fit_transform(X[:, 0])
one_hot_encoder = OneHotEncoder(categorical_features=[0])
X = one_hot_encoder.fit_transform(X).toarray()
label_encoder_purchased = LabelEncoder()
Y = label_encoder_purchased.fit_transform(Y)


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)