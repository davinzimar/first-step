# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 21:08:37 2020

@author: VIN-PC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from IPython.html import widgets 
from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

path = (r'E:\Davin\Python\Data Analyst\Source\automobileEDA.csv')
df = pd.read_csv(path)
#df.head()

# Simpan file menjadi format .csv
#df.to_csv('punya davin.csv')

# Menampilkan data yang hanyan berisi angka
df = df._get_numeric_data()
df.head()

# Fungsi untuk plotting
def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    
    # Ukuran frame yang akan dihasilkan
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    # Plot garis
    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)
    
    # Labeling
    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    plt.show()
    plt.close()
    
def PollyPlot(xtrain, xtest, y_train, y_test, lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    
    #training data 
    #testing data 
    # lr:  linear regression object 
    #poly_transform:  polynomial transformation object 
 
    xmax=max([xtrain.values.max(), xtest.values.max()])

    xmin=min([xtrain.values.min(), xtest.values.min()])

    x=np.arange(xmin, xmax, 0.1)


    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()
    
# Target
y_data = df['price']
# Bahan
x_data=df.drop('price',axis=1)

# Split data menjadi train dan test
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=1)

print("Banyak sampel test:", x_test.shape[0])
print("Banyak sampel training:",x_train.shape[0])

# Linear regression (Horsepower sbg contoh)
lre=LinearRegression()
lre.fit(x_train[['horsepower']], y_train)
skor_tes_lre=lre.score(x_test[['horsepower']], y_test)
skor_train_lre=lre.score(x_train[['horsepower']], y_train)

print(skor_tes_lre)
print(skor_train_lre)

# Cross Validation Score (Horsepower sbg contoh)
Rcross = cross_val_score(lre, x_data[['horsepower']], y_data, cv=4)
print(Rcross)
print("Rata-rata lipatannya adalah:", Rcross.mean(), "Standar deviasinya adalah:" , Rcross.std())