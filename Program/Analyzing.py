# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 11:08:12 2020

@author: VIN-PC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

path = (r'E:\Davin\Python\Data Analyst\Source\automobileEDA.csv')
df = pd.read_csv(path)
df.head()

#Print tipe data di masing-masing kolom
print(df.dtypes)

#besarnya korelasi antara ukuran mesin dengan harga
corr_UkMesin_harga = df[["engine-size", "price"]].corr()

# Plot ukuran mesin menjadi prediksi harga
#sns.regplot(x="engine-size", y="price", data=df)
#plt.ylim(0,)

#Deskripsi setiap kolom
deskripsi = df.describe()
deskripsi_all = df.describe(include='object')

#Menghitung isi baris pada kolom tertentu
isi_baris = df['drive-wheels'].value_counts()

#membuat grup dari beberapa kolom dan melihat harga rata2
df_grup_satu = df[['drive-wheels','body-style','price']]
df_grup_satu = df_grup_satu.groupby(['body-style'],as_index=False).mean()

#membuat grup dari beberapa kolom dan melihat harga tertinggi
df_grup_dua = df[['body-style','price']]
df_grup_dua = df_grup_dua.groupby(['body-style'],as_index=False).min()