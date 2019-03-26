from sklearn import tree
from sklearn import preprocessing
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
#import numpy as np
#import os

df1 = pd.read_excel('DATA/acceleration_data_1_CRACK_0.0mm.xlsx')
df2 = pd.read_excel('DATA/acceleration_data_1_CRACK_1.5mm.xlsx')
df3 = pd.read_excel('DATA/acceleration_data_1_CRACK_3.0mm.xlsx')
df4 = pd.read_excel('DATA/acceleration_data_1_CRACK_4.5mm.xlsx')
df1 = df1.T
df1['crack'] = 0.0
df2 = df2.T
df2['crack'] = 1.5
df3 = df3.T
df3['crack'] = 3.0
df4 = df4.T
df4['crack'] = 4.5
df = df1.append(df2.append(df3.append(df4)))
df = df.drop(['time_v'])
df = df.dropna()

X = df.drop('crack',axis=1)
y = df['crack']


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
model = tree.DecisionTreeClassifier()
lab_enc = preprocessing.LabelEncoder()
encode =  lab_enc.fit_transform(y_train)
utils.multiclass.type_of_target(y_train.astype('int'))
utils.multiclass.type_of_target(encode)

model.fit(X_train,y_train)
y_predict = model.predict(X_test)
accuracy_score(y_test, y_predict)

print(model)