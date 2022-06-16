

from numpy import mean
from numpy import std
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
# X_train is an array of 60000 sub arrays where
# each sub array has 28x28 values
#load dataset
def load_dataset():
    (trainX,trainY),(testX,testY)=fashion_mnist.load_data()
    ## reshape dataset to have a single channel
    trainX=trainX.reshape((trainX.shape[0],28,28,1))
    testX=testX.reshape((testX.shape[0],28,28,1))
    ## one hot encode labels
    trainY=to_categorical(trainY)
    testY=to_categorical(testY)
    return trainX,trainY,testX,testY

## scale pixels between [0,1] for [0,255] color values
def prep_pixels(train,test):
    ## convert integers to floats
    train_norm=train.astype('float32')
    test_norm=test.astype('float32')
    ## normalize
    train_norm=train_norm/255.0
    test_norm=test_norm/255.0
    return train_norm,test_norm

## define model
def define_model():
    model=Sequential()
    model.add(Conv2D(32,(3,3),activation='relu',kernel_initializer='he_uniform',input_shape=(28,28,1)))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(100,activation='relu',kernel_initializer='he_uniform'))
    model.add(Dense(10,activation='softmax'))
    opt=SGD(learning_rate=0.01,momentum=0.9)
    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
    return model

## evaluate model using k fold
def evaluate_model(dataX,dataY,n_folds=5):
    scores, histories=list(), list()
    kfold=KFold(n_splits=n_folds,shuffle=True,random_state=1)
    ##enumerate splits
    for train_ix,test_ix in kfold.split(dataX):
        model=define_model()
        ##select rows for train and test
        trainX,trainY,testX,testY=dataX[train_ix],dataY[train_ix],dataX[test_ix],dataY[test_ix]
        ##fit model
        history=model.fit(trainX,trainY,epochs=10, batch_size=32,validation_data=(testX,testY),verbose=0)
        _,acc=model.evaluate(testX,testY,verbose=0)
        print('%.3f' % (acc*100))
        ##append scores
        scores.append(acc)
        histories.append(history)
    return scores,histories

## plot diagnostic curves of training histories
def summarize_diagnostics(histories):
    for i in range(len(histories)):
        ##plot loss
        plt.subplot(211)
        plt.title('Cross entropy loss')
        plt.plot(histories[i].history['loss'],color='blue',label='train')
        plt.plot(histories[i].history['val_loss'],color='orange',label='test')
        ##plot accuracy
        plt.subplot(212)
        plt.title('Classification accuracy')
        plt.plot(histories[i].history['accuracy'],color='blue',label='train')
        plt.plot(histories[i].history['val_accuracy'],color='orange',label='test')  
    plt.show()

##summarize model performance
def summarize_performance(scores):
    print('Accuracy: mean=%.3f std=%.3f,n=%d'%(mean(scores)*100, std(scores)*100),len(scores))

      
## run the test_harness for evaluating a model
def run_test_harness():
    trainX,trainY,testX,testY=load_dataset()
    trainX,trainY=prep_pixels(trainX,trainY)
    scores,histories=evaluate_model(trainX,trainY)
    summarize_diagnostics(histories)
    summarize_performance(scores)

run_test_harness()



##model.save('modelname.h5')
## load and prepare an image for prediction
from keras.preprocessing.image import load_img,img_to_array 
# def load_img(filename):
    # img=load_img(filename,grayscale=True,target_size=(28,28))
    # img=img_to_array(img)
    # img=img.reshape(1,28,28,1)
    # img=img.astype('float32')
    # img=img/255.0
    # return img

# result=model.predict_classes(img)







































