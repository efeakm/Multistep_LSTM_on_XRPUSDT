# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 18:42:36 2022

@author: Efeakm
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt

SEED = 145
os.environ['PYTHONHASHSEED']=str(SEED)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # new flag present in tf 2.0+
np.random.seed(SEED)
tf.random.set_seed(SEED)


###INPUT PARAMETERS
#================================================================================
TAKE_PROFIT = 0.05
STOP_LOSS = 0.05




###PREDICTIONS
#==============================================================================

#Import Datas
df = pd.read_csv('Data/preprocessed_df.csv')
df = df.set_index('Time')
lstm_x_test = np.load("Data/lstm_x_test.npy")
lstm_y_test = np.load("Data/lstm_y_test.npy")


#Load Model
model1 = tf.keras.models.load_model('lstm_model')


logits = model1.predict(lstm_x_test, batch_size = 16)
lstm_predictions = tf.nn.sigmoid(logits).numpy().argmax(axis = 1)


print('predicted class weights:',
      pd.Series(lstm_predictions).value_counts(normalize = True))



###ANALYZE AND PLOT RESULTS
#==============================================================================

lstm_results = pd.DataFrame(lstm_predictions,columns = ['pred'])
lstm_results.index = df.index[-len(lstm_results):]
lstm_results = lstm_results.join(df['Close'])



lstm_results['volatility'] = np.log1p(((lstm_results['Close'].shift(-12) - lstm_results['Close']) /
                              lstm_results['Close']).abs())


lstm_results['volatility'] = MinMaxScaler().fit_transform(
    lstm_results['volatility'].values.reshape(-1,1))*3


mask = lstm_results['pred'] == 1
long_signals = lstm_results[mask].copy()
mask = lstm_results['pred'] == 2
short_signals = lstm_results[mask].copy()


plt.style.use('seaborn-whitegrid')
fig,ax = plt.subplots(1,1,figsize = (10,5))
ax.scatter(pd.to_datetime(lstm_results.index), lstm_results['Close'],
         color = plt.cm.Reds(lstm_results['volatility']), s = 10)

ax.scatter(pd.to_datetime(long_signals.index), long_signals['Close']-0.05, s = 5, color = 'blue')
ax.scatter(pd.to_datetime(short_signals.index), short_signals['Close']+0.05, s = 5, color = 'purple')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title('XRPUSDT Volatility Signals', fontweight = 'bold')

fig.savefig('plot.png')




###BACKTESTING
#=============================================================================
df_backtest = lstm_results.drop('volatility',axis = 1)
df_backtest = df_backtest.join(df[['High','Low']])


results = []
for i in range(len(df_backtest)):
    
    #Long signal
    if df_backtest.iloc[i,0] == 1:
        tp_price = df_backtest.iloc[i,1] * (1+TAKE_PROFIT)
        sl_price = df_backtest.iloc[i,1] * (1-STOP_LOSS)
        
        mask = (df_backtest.iloc[i:,:]['High'] > tp_price).to_list()
        try:
            tp_date = mask.index(True)
        except:
            tp_date = np.inf
            
            
        mask = (df_backtest.iloc[i:,:]['Low'] <= sl_price).to_list()
        try:
            sl_date = mask.index(True)
        except:
            sl_date = np.inf
        
        if tp_date < sl_date:
            results.append(True)
        elif tp_date > sl_date:
            results.append(False)
        else:
            print('error')
            
            
    #Short signal
    if df_backtest.iloc[i,0] == 2:
        tp_price = df_backtest.iloc[i,1] * (1-TAKE_PROFIT)
        sl_price = df_backtest.iloc[i,1] * (1+STOP_LOSS)
        
        
        mask = (df_backtest.iloc[i:,:]['Low'] < tp_price).to_list()
        try:
            tp_date = mask.index(True)
        except:
            tp_date = np.inf
            
            
        mask = (df_backtest.iloc[i:,:]['High'] >= sl_price).to_list()
        try:
            sl_date = mask.index(True)
        except:
            sl_date = np.inf
        
        if tp_date < sl_date:
            results.append(True)
        elif tp_date > sl_date:
            results.append(False)
        else:
            print('error at',i)
    else:
        pass
        

print('BACKTESTING RESULTS:')
print(pd.Series(results).value_counts())

