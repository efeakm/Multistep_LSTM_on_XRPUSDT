# -*- coding: utf-8 -*-
"""
Created on Mon May 30 18:53:05 2022

@author: Efeakm
"""
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Bidirectional,Dropout
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

SEED = 145
os.environ['PYTHONHASHSEED']=str(SEED)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # new flag present in tf 2.0+
np.random.seed(SEED)
tf.random.set_seed(SEED)



###INPUTS
#==============================================================================
TIMESTEP = 6




#Import Datas
df = pd.read_csv('Data/preprocessed_df.csv')
df = df.set_index('Time')
lstm_x_test = np.load("Data/lstm_x_test.npy")
lstm_y_test = np.load("Data/lstm_y_test.npy")
lstm_x_val = np.load("Data/lstm_x_val.npy")
lstm_y_val = np.load("Data/lstm_y_val.npy")
lstm_x_train = np.load("Data/lstm_x_train.npy")
lstm_y_train = np.load("Data/lstm_y_train.npy")
test_x = pd.read_csv('Data/test_x.csv')
test_y = pd.read_csv('Data/test_y.csv')

print(pd.Series(lstm_y_train).value_counts(normalize=True))


#Calculate class weights
class_weight_temp = class_weight.compute_class_weight(class_weight = 'balanced',
                                          classes = [0,1,2], y = lstm_y_train)
class_weights = dict()
for i in range(3):
    class_weights[i] = round(class_weight_temp[i],2)

print('Class Weights:',class_weight_temp)
del(class_weight_temp)



###LSTM MODEL
#==================================================================================
N_FEATURES = len(test_x.columns)


keras.backend.clear_session()
model1 = Sequential([
        Bidirectional(LSTM(128, activation = 'tanh',
            input_shape = (TIMESTEP,N_FEATURES), dropout = 0.2, 
            return_sequences = True)),
        Bidirectional(LSTM(16, activation = 'tanh',
             input_shape = (TIMESTEP,N_FEATURES), dropout = 0.2, 
             return_sequences = False)),
        # Dense(10, activation = 'relu'),
        # Dropout(0.2),
        Dense(3)
        ])



model1.compile(optimizer = 'Adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                metrics = ['accuracy']
              )



history = model1.fit(x = lstm_x_train, y = lstm_y_train, batch_size = 32,
                    epochs = 80, validation_data = [lstm_x_val, lstm_y_val],
                    class_weight = class_weights,
                    callbacks = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                            patience = 7,restore_best_weights = True),
          )


model1.save('lstm_model')
    
    
#Plot Train and Val loss
plt.plot(history.history['loss'][5:], label ='loss')
plt.plot(history.history['val_loss'][5:], label = 'val_loss')
plt.legend()



import winsound
winsound.Beep(5000,1000)


