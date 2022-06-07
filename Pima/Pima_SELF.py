
import numpy as np
import pandas as pd
# 
# dataset=np.loadtxt('pima-indians-diabetes.data.csv',delimiter=',')
# print(dataset.shape)
# print(dataset[5,:])
dataset2=pd.read_csv('pima-indians-diabetes.data.csv')
X=dataset2.iloc[:,0:8].astype(float)
y=dataset2.iloc[:,8].astype(float)
# x=dataset[:,0:8]
# y=dataset[:,8]
# print(X)
# print(x)

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

def create_model():
    model=Sequential()
    model.add(Dense(units=30,input_dim=8,activation='relu'))
    model.add(Dense(units=20,activation='relu'))
    model.add(Dense(units=1,activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model

model=KerasClassifier(build_fn=create_model,epochs=150,batch_size=10)
kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=None)

results=cross_val_score(model,X,y,cv=kfold)
print(results.mean())





 
























