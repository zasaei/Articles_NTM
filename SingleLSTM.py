# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 10:24:59 2019

@author: F1-PC
"""
def plot_history(net_history):
    history = net_history.history
    import matplotlib.pyplot as plt
    losses = history['loss']
    val_losses = history['val_loss']
    accuracies = history['acc']
    val_accuracies = history['val_acc']
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.plot(val_losses)
    plt.legend(['loss', 'val_loss'])
    
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(accuracies)
    plt.plot(val_accuracies)
    plt.legend(['acc', 'val_acc'])

#from __future__ import print_function
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding                           
from keras.layers import LSTM
#from keras.datasets import imdb


#from keras.models import Sequential
from keras.layers import  Dropout
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy

max_features = 500
# cut texts after this number of words (among top max_features most common words)
maxlen = 80
#batch_size = 32

import pandas 
import numpy as np
#import matplotlib as plt
#%matplotlib inline
i_path="C:/data/1N.xlsx"
i_i=pandas.ExcelFile(i_path)
df1=np.array(i_i.parse('Sheet2'))
X_train=df1[20621:58919,0:27]
Y_train=df1[20621:58919,27:29]

X_test=df1[0:20621,0:27]
Y_test=df1[0:20621,27:29]

#validdata=df1[45000: ,:]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
Y_train = Y_train.astype('float32')
Y_test = Y_test.astype('float32')


print('Loading data...') 
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('x_train shape:', X_train.shape)
print('x_test shape:', X_test.shape)

print('Build model...')

myModel = Sequential()
myModel.add(Embedding(max_features,120))
#myModel.add(LSTM(output_dim=100, dropout=0.2, recurrent_dropout=0.2))

#myModel.add(LSTM(output_dim=80, activation='sigmoid', inner_activation='hard_sigmoid'))
myModel.add(LSTM(80,dropout=0.2, recurrent_dropout=0.2))
 
myModel.add(Dropout(2))
#myModel.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
myModel.add(Dense(2, activation='softmax'))

'''
myModel = Sequential()
myModel.add(Embedding(max_features,50))
myModel.add(LSTM(50, dropout=0.2, recurrent_dropout=0.2))
myModel.add(Dropout(20))
#myModel.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
#myModel.add(Dropout(20))
#myModel.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
myModel.add(Dense(2, activation='softmax'))

# try using different optimizers and different optimizer configs
'''
'''
myModel = Sequential()
myModel.add(Dense(2000, activation='relu', input_shape=(27,)))
myModel.add(Dropout(20))
myModel.add(Dense(1000, activation='relu'))
myModel.add(Dropout(20))
myModel.add(Dense(200, activation='relu'))
myModel.add(Dropout(20))
myModel.add(Dense(50, activation='relu'))
myModel.add(Dropout(20))
myModel.add(Dense(2, activation='softmax'))
'''
myModel.summary()
'''
myModel.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
'''
#myModel.compile(optimizer=SGD(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
myModel.compile(optimizer=SGD(lr=0.0001), loss=categorical_crossentropy, metrics=['accuracy'])
#score = myModel.evaluate(X_test, Y_test, batch_size=16)
#=================================================

# Train our model
network_history = myModel.fit(X_train, Y_train, batch_size=32, epochs=50, validation_split=0.2)
plot_history(network_history)

# Evaluation
test_loss, test_acc = myModel.evaluate(X_test, Y_test)
test_labels_p = myModel.predict(X_test)
import numpy as np
test_labels_p = np.argmax(test_labels_p, axis=1)


# Change layers config
myModel.layers[0].name = 'Layer_0'
myModel.layers[0].trainable = False
myModel.layers[0].get_config()
