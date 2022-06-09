# -*- coding: utf-8 -*-
"""
Created on Fri May 27 11:43:57 2022

@author: Efeakm
"""

import pandas as pd
import numpy as np
import os
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

SEED = 145
os.environ['PYTHONHASHSEED']=str(SEED)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # new flag present in tf 2.0+
np.random.seed(SEED)



###INPUT
#==============================================================================
TIMESTEP = 6
TEST_SIZE = 1000



###INITIALIZATION
#==============================================================================
df = pd.read_csv('Data/preprocessed_df.csv')
df = df.set_index('Time')



x_train = df.drop('volatility',axis = 1).iloc[:-TEST_SIZE,:]
y_train = df['volatility'].iloc[:-TEST_SIZE].copy()


test_x = df.drop('volatility',axis = 1).iloc[-TEST_SIZE:,:]
test_y = df['volatility'].iloc[-TEST_SIZE:].copy()


#Standardize Variables
scaler = StandardScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train))
test_x = pd.DataFrame(scaler.transform(test_x))
#Save scaler to disk
pickle.dump(scaler, open('scaler.pkl','wb'))



print('Train Class Weights:')
print(y_train.value_counts(normalize=True))
print()
print('Test Class Weights:')
print(test_y.value_counts(normalize=True))




### RESHAPE FOR LSTM WITH TIMESTEPS
# ===================================================================================

# Input Shape (sample_size,TIMESTEP,N_FEATURES)
# Output Shape = (sample_size,1)
N_FEATURES = len(x_train.columns)


def lstm_reshape(x,y, TIMESTEP, N_FEATURES):

    lstm_x = pd.DataFrame()
    for i in range(len(x)- TIMESTEP):
        lstm_x = pd.concat([lstm_x,x.iloc[i:i+TIMESTEP,:]],axis = 0)
        if i % 1000 == 0:
            print(i)
    
    lstm_x = np.reshape(lstm_x.values,(len(x)-TIMESTEP, TIMESTEP, N_FEATURES))
    gc.collect()
    return lstm_x, y[TIMESTEP:].reset_index(drop=True)


lstm_x_train,lstm_y_train = lstm_reshape(x_train, y_train, TIMESTEP, N_FEATURES)
lstm_x_test, lstm_y_test = lstm_reshape(test_x, test_y, TIMESTEP, N_FEATURES)




train_idx,val_idx = train_test_split(lstm_y_train, test_size = 0.2, random_state = 145,
                       stratify = lstm_y_train)
lstm_x_val= lstm_x_train[val_idx.index]
lstm_y_val = lstm_y_train[val_idx.index]
lstm_x_train= lstm_x_train[train_idx.index]
lstm_y_train = lstm_y_train[train_idx.index]

print(lstm_y_train.value_counts(normalize = True))
print(lstm_y_val.value_counts(normalize = True))



np.save("Data/lstm_x_test.npy", lstm_x_test)
np.save("Data/lstm_y_test.npy", lstm_y_test)
np.save("Data/lstm_x_val.npy", lstm_x_val)
np.save("Data/lstm_y_val.npy", lstm_y_val)
np.save("Data/lstm_x_train.npy", lstm_x_train)
np.save("Data/lstm_y_train.npy", lstm_y_train)
test_x.to_csv('Data/test_x.csv')
test_y.to_csv('Data/test_y.csv')


import winsound
winsound.Beep(5000,1000)


