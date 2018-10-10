#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 18:18:58 2018

@author: colosu
"""


#Libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn as skl
from sklearn import preprocessing
import matplotlib.pyplot as plt


# Read Dataset
dataset = pd.read_csv('./Adult_Income.txt',delimiter=',',header=0)
#dataset

#0 for <=50K
#1 for >50K

# Header
header = []
for row in dataset:
    header.append(row)
header

encoders = {}
pos = []

for column in dataset.columns:
    if dataset.dtypes[column] == 'object':
        count = 0
        for elem in dataset[column]:
            if elem == " ?":
                dataset = dataset.drop(count)
                pos.append(count)
            count += 1
            while pos.__contains__(count):
                count += 1
        #encoders[column] = preprocessing.LabelEncoder()
        #dataset[column] = encoders[column].fit_transform(dataset[column].astype('str'))
        


print(dataset.describe()) #Descripción de los datos

correlation=dataset.corr() #Correlation Matrix
print(correlation)

dataset = dataset.drop(" fnlwgt", axis=1)
dataset = dataset.drop(" education", axis=1)
dataset = pd.get_dummies(dataset)

print(dataset.describe()) #Descripción de los datos

correlation=dataset.corr() #Correlation Matrix
print(correlation)

iqr = dataset.quantile(q=0.75, axis=0)-dataset.quantile(q=0.25, axis=0)
q1 = dataset.quantile(q=0.25, axis=0) - 1.5*iqr
q3 = dataset.quantile(q=0.75, axis=0) + 1.5*iqr

i = 0
poss = []
for elem in dataset[header[4]] < q1[" education-num"]: #Son 196
    if elem:
        poss.append(i)
    i += 1
    while pos.__contains__(i):
        i += 1

for i in poss:
    dataset = dataset.drop(i)
    

"""
# Display the correlation matrix with a specified figure number and a bluescale
# colormap
plt.figure()
plt.matshow(correlation, fignum=1, cmap=plt.cm.Blues)
plt.ylabel("Attribute Index")
plt.show()

#### Scatter Matrix Plot

plt.figure()
from pandas.tools.plotting import scatter_matrix
scatter_matrix(dataset, alpha=0.3, figsize=(20, 20), diagonal='kde')
plt.show()

#### Histogram Matrix Plot

plt.figure()
dataset.hist(xlabelsize=0.5, ylabelsize=0.2,figsize=(10,10))
plt.xlabel("Data")
plt.show()
"""

"""
### Histogram of Alcohol variable

plt.figure(figsize=(8,4))
plt.hist(dataset[dataset.Wine==1].Alcohol, 30, facecolor='r')
plt.hist(dataset[dataset.Wine==2].Alcohol, 30, facecolor='g')
plt.hist(dataset[dataset.Wine==3].Alcohol, 30, facecolor='b')
plt.title('Histogram of Alcohol')
plt.legend(['Wine A','Wine B','Wine C'])
plt.xlabel("Alcohol")
plt.grid(True)
plt.show()

### Scatter Plot Alcohol vs Ash

plt.figure()
plt.scatter(dataset[dataset.Wine==1].Alcohol,dataset[dataset.Wine==1].Ash, color='red')
plt.scatter(dataset[dataset.Wine==2].Alcohol,dataset[dataset.Wine==2].Ash, color='blue')
plt.scatter(dataset[dataset.Wine==3].Alcohol,dataset[dataset.Wine==3].Ash, color='green')
plt.title('Scatter Plot: Alcohol vs Ash')
plt.xlabel('Alcohol')
plt.ylabel('Ash')
plt.legend(['A','B','C'])
plt.show()
"""