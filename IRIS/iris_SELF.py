
from pyexpat import model
from unittest import result
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
# from numpy import loadtxt

#load dataset
dataset=pd.read_csv('iris.csv')
X=dataset.iloc[:,0:-1].astype(float).values
y=dataset.iloc[:,-1].astype(str).values
# print(X)

# dataframe = pd.read_csv("iris.csv", header=None)
# dataset = dataframe.values
# X = dataset[:,0:4].astype(float)
# y = dataset[:,4]

#use labelencoder for the output labels
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
le.fit(y)
encoded_y=le.transform(y)
# print(encoded_y)
#one hot encode y
dummy_y=np_utils.to_categorical(encoded_y)
# print(dummy_y)

#build model
from keras import Sequential
from keras.layers import Dense
from sklearn.model_selection import cross_val_score

def create_model():
    model=Sequential()
    model.add(Dense(units=50,input_dim=4,activation='relu'))
    model.add(Dense(units=30,activation='relu'))
    model.add(Dense(units=3,activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
    return model

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
estimator=KerasClassifier(build_fn=create_model,epochs=200, batch_size=10)
kfold=KFold(n_splits=10,shuffle=True,random_state=None)
results=cross_val_score(estimator,X,dummy_y,cv=kfold)

print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))





















# X=dataset.iloc[:,0:-1].values
# y=dataset.iloc[:,-1].values















# from sklearn.preprocessing import LabelEncoder
# lc=LabelEncoder()
# lc.fit(y)
# y_encoded=lc.transform(y)
# y_dummy=np_utils.to_categorical(y_encoded)
# print(y_dummy)

# from sklearn.model_selection import train_test_split
# X_train,X_test,y_train,y_test=train_test_split(X,y_dummy,test_size=0.2,random_state=0)
# print(y_train)
# 
# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Dense
# 
# model=Sequential()
# model.add(Dense(units=30,activation='relu'))
# model.add(Dense(units=20,activation='relu'))
# model.add(Dense(units=10,activation='relu'))
# model.add(Dense(units=1,activation='softmax'))
# 
# model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
# 
# model.fit(X_train,y_train,epochs=50,batch_size=30)
# 
# 


