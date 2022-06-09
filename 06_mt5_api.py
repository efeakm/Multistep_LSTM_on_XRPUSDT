# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 13:51:22 2022

@author: Efeakm
"""

import MetaTrader5 as mt5
import requests
import pandas as pd
import tensorflow as tf
from feature_engineering import feature_engineering
import pickle

###INPUT PARAMETERS
#==============================================================================



#Account parameters
ACCOUNT = 0000000000
PASSWORD = ""
SERVER = ""


#Trading Parameters
POSITION_SIZE = 100.0
TAKE_PROFIT = 0.06
STOP_LOSS = 0.04
SYMBOL = 'XRPUSD'
SYMBOL_BINANCE = 'XRPUSDT'
INTERVAL = '1h'



###GET THE DATA
#Get the data
def get_ohlc_data(SYMBOL, INTERVAL, ENDTIME = None):
    BIN_API = ''
    URL = 'https://api.binance.com'
    headers = {
        'X-MBX-APIKEY': BIN_API
    }

    PATH = '/api/v3/klines'
    params = {
        'symbol': SYMBOL,
        'interval':INTERVAL,
        'limit': 220,
        'endTime': ENDTIME
    }       
    
    
    r = requests.get(URL+PATH, headers=headers, params=params)
    df = pd.DataFrame(r.json()).iloc[:,:6]
    df.columns = ['Time','Open','High','Low','Close','Volume']
    
    df['Time'] = pd.to_datetime(df['Time'], unit = 'ms')
    df[['Open','High','Low','Close','Volume']] = df[['Open','High','Low','Close','Volume']].astype('float')
    
    return df


df_pred = get_ohlc_data(SYMBOL_BINANCE, INTERVAL, ENDTIME = None)
df_pred = df_pred.set_index('Time')

#Feature Engineering
df_pred = feature_engineering(df_pred)
df_pred = df_pred.iloc[-6:,:]



#Scaling
scaler = pickle.load(open('scaler.pkl','rb'))
scaler.transform(df_pred)





###PREDICTION
#==================================================================================
model = tf.keras.models.load_model('lstm_model')




###MT5 INITIALIZATION
#==============================================================================
mt5.initialize(login=ACCOUNT, password=PASSWORD, server=SERVER)
if not mt5.initialize():
    print("initialize() failed, error code =",mt5.last_error())
if mt5.login(ACCOUNT, PASSWORD, server = SERVER):
    print("mt5 connected")



###BUY ORDER
#==============================================================================

position_entry = round(mt5.symbol_info_tick(SYMBOL).ask ,3)
position_sl = round(mt5.symbol_info_tick(SYMBOL).bid * (1 - STOP_LOSS),3)
position_tp = round(mt5.symbol_info_tick(SYMBOL).bid * (1 + TAKE_PROFIT),3)


###Send Buy Order
order = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": SYMBOL,
    "volume": POSITION_SIZE,
    "type": mt5.ORDER_TYPE_BUY,
    "price": mt5.symbol_info_tick(SYMBOL).ask,
    "sl": position_sl,
    "tp": position_tp,
    "deviation": 10,
    "magic": 234000,
    "comment": "TEST BUY ORDER",
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": mt5.ORDER_FILLING_FOK,
}
result = mt5.order_send(order)
print(result)



###SELL ORDER
#=============================================================================================

position_entry = round(mt5.symbol_info_tick(SYMBOL).bid, 3)
position_sl = round(mt5.symbol_info_tick(SYMBOL).ask * (1 + STOP_LOSS ),3)
position_tp = round(mt5.symbol_info_tick(SYMBOL).ask * (1 - TAKE_PROFIT),3)


#Send sell order
order = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": SYMBOL,
    "volume": POSITION_SIZE,
    "type": mt5.ORDER_TYPE_SELL,
    "price": mt5.symbol_info_tick(SYMBOL).ask,
    "sl": position_sl,
    "tp": position_tp,
    "deviation": 10,
    "magic": 234000,
    "comment": "ATR sell order",
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": mt5.ORDER_FILLING_FOK,
}
result = mt5.order_send(order)
print(result)



















