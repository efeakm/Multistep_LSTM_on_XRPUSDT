# -*- coding: utf-8 -*-
"""
Created on Thu May 26 18:58:12 2022

@author: Efeakm
"""

import pandas as pd
from feature_engineering import feature_engineering


BEFORE_VOLATILITY_PERIOD = 6



df = pd.read_csv('Data/XRPUSDT_1h.csv')


###VOLATILITY
#===================================================================================

#Volatility i.e. 5 percent change in prices
df['close_12h'] = df['Close'].shift(-12)
df['volatility_temp'] = (df['close_12h'] - df['Close']) / df['Close']
mask = df['volatility_temp'] >= 0.05
df.loc[mask,'volatility'] = 1
mask = df['volatility_temp'] <= - 0.05
df.loc[mask,'volatility'] = 2
df['volatility'] = df['volatility'].fillna(0)
df = df.drop(['volatility_temp','close_12h'], axis = 1)



###INDICATORS
#===================================================================================
df = feature_engineering(df)



###SAVE OUTPUT
#======================================================================
mask = df.isna().sum(axis = 1) == 0
df = df[mask]


df.to_csv('Data/preprocessed_df.csv',index = False)




