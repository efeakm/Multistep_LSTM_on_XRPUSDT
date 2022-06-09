# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 14:43:17 2022

@author: Efeakm
"""
import requests
import pandas as pd
import time
import os

###PARAMETERS
#================================================================================
SYMBOL_LIST = ['XRPUSDT']
INTERVAL = '1h'
DOWNLOAD_PERIOD_LOOP_NUMBER = 100



BIN_API = ''


URL = 'https://api.binance.com'
headers = {
    'X-MBX-APIKEY': BIN_API
}

PATH = '/api/v3/klines'



#Get the data
def get_ohlc_data(SYMBOL, INTERVAL, ENDTIME = None):
    

    params = {
        'symbol': SYMBOL,
        'interval':INTERVAL,
        'limit': 1000,
        'endTime': ENDTIME
    }       
    
    
    r = requests.get(URL+PATH, headers=headers, params=params)
    df = pd.DataFrame(r.json()).iloc[:,:6]
    df.columns = ['Time','Open','High','Low','Close','Volume']
    ENDTIME = df.iloc[0,0]
    
    df['Time'] = pd.to_datetime(df['Time'], unit = 'ms')
    df[['Open','High','Low','Close','Volume']] = df[['Open','High','Low','Close','Volume']].astype('float')
    
    return df, ENDTIME



### GET MONTHLY DATA
#===========================================================================

for symbol in SYMBOL_LIST:
    
    df = pd.DataFrame(columns = ['Time','Open','High','Low','Close','Volume'])
    df_temp, endtime_new = get_ohlc_data(symbol,INTERVAL)
    df = pd.concat([df,df_temp], axis = 0)
    
    for i in range(DOWNLOAD_PERIOD_LOOP_NUMBER):
        
        time.sleep(0.5)
        endtime_prev = endtime_new
        df_temp, endtime_new = get_ohlc_data(symbol,INTERVAL, endtime_prev)
        
        #Stop if there is no more data available
        if endtime_new == endtime_prev:
            break
        
        print(symbol,'downloaded till', pd.to_datetime(endtime_new, unit = 'ms').date())
        df = pd.concat([df,df_temp], axis = 0)

            
    
    df = df.sort_values('Time', ignore_index=True)
    df = df.drop_duplicates('Time')
    df.to_csv(f'Data/{symbol}_{INTERVAL}.csv', index = False)
    print(f'{symbol}_{INTERVAL} downloaded')














