# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 10:23:33 2020

@author: VIN-PC
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns

path = (r'E:\Davin\Python\Data Analyst\Source\automobileEDA.csv')
df = pd.read_csv(path)
#df.head(5)

lm = LinearRegression()

X = df[['city-mpg']]
Y = df['price']

lm.fit(X,Y)

Yhat=lm.predict(X)
Yhat[0:5]

intercept = lm.intercept_
slope = lm.coef_

width = 6
height = 5
plt.figure(figsize=(width, height))
sns.regplot(x="city-mpg", y="price", data=df)
plt.ylim(0,)