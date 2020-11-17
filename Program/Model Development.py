# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 16:12:39 2020

@author: VIN-PC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

path = (r'E:\Davin\Python\Data Analyst\Source\automobileEDA.csv')
df = pd.read_csv(path)
df.head()

# ---Single Linear Regression---

#Variabel
X = df[['highway-mpg']]
Y = df[['price']]

#Fit the linear model
lm = LinearRegression()
lm.fit(X,Y)

#Output Prediksinya
Yhat=lm.predict(X)
Yhat[0:5]

#Nilai Intersep dan slopenya
intercept1 = lm.intercept_
slope1 = lm.coef_

# --------------------------------

# ---Multiple Linear Regression---

#Variabelnya banyak
XX = df[['highway-mpg','city-mpg','engine-size','horsepower']]

#Fit the linear model
lm2 = LinearRegression()
lm2.fit(XX, df['price'])

#Output Prediksinya
Yhat2=lm2.predict(XX)
Yhat2[0:5]

#Nilai Intersep dan slopenya
intercept2 = lm2.intercept_
slope2 = lm2.coef_
# --------------------------------

# ---Plot Regresi---

#Single Linear Regression Plot
width = 6
height = 5
plt.figure(figsize=(width, height))
sns.regplot(x="city-mpg", y="price", data=df)
plt.ylim(0,)

#Multiple Linear Regression Plot
plt.figure(figsize=(width, height))


ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Yhat, hist=False, color="b", label="Fitted Values" , ax=ax1)


plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()
plt.close()
# ----------------------------------

