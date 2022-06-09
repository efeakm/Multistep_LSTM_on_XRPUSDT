import talib
def feature_engineering(df_input):
    
    df = df_input.copy()
    openn = df['Open'].copy()
    high = df['High'].copy()
    low = df['Low'].copy()
    close = df['Close'].copy()
    volume = df['Volume'].copy()
    
    
    #SMA
    for i in [20,50,100,200]:
        df[f'ma_{i}'] = talib.MA(df['Close'],timeperiod = i) / df['Close']
    
    #BB
    df['bb_up']= talib.BBANDS(df['Close'],timeperiod = 20)[0] / df['Close']
    df['bb_down']= talib.BBANDS(df['Close'],timeperiod = 20)[2] / df['Close']
    df['bb_range'] = df['bb_up'] / df['bb_down']
    
    #ATR
    df['atr'] = talib.NATR(df['High'],df['Low'],df['Close'])
    
    
    #And more
    df['adx'] = talib.ADX(high, low, close, timeperiod=14)
    df['dx'] = talib.DX(high, low, close, timeperiod=14)
    df['macd_1'] = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)[0]
    df['macd_2'] = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)[1]
    df['macd_3'] = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)[2]
    df['mfi'] = talib.MFI(high, low, close, volume, timeperiod=14)
    df['ad'] = talib.AD(high, low, close, volume)
    df['ht_cycle'] = talib.HT_DCPHASE(close)
    df['lin_slope'] = talib.LINEARREG_SLOPE(close, timeperiod=14)
    df['ts_forecast'] = talib.TSF(close, timeperiod=14)
    df['blackcrow'] = talib.CDL3BLACKCROWS(openn, high, low, close)
    df['whitesol'] = talib.CDL3WHITESOLDIERS(openn, high, low, close)
    df['breakaway'] = talib.CDLBREAKAWAY(openn, high, low, close)
    df['counteratt'] = talib.CDLCOUNTERATTACK(openn, high, low, close)
    df['doji'] = talib.CDLDOJI(openn, high, low, close)
    df['engulf'] = talib.CDLENGULFING(openn, high, low, close)
    df['hammer'] = talib.CDLHAMMER(openn, high, low, close)
    df['hanging'] = talib.CDLHANGINGMAN(openn, high, low, close)
    df['inneck'] = talib.CDLINNECK(openn, high, low, close)
    df['kicking'] = talib.CDLKICKINGBYLENGTH(openn, high, low, close)
    df['longlegdoji'] = talib.CDLLONGLEGGEDDOJI(openn, high, low, close)
    df['piercing'] = talib.CDLPIERCING(openn, high, low, close)
    df['spinningtop'] = talib.CDLSPINNINGTOP(openn, high, low, close)
    df['thrusting'] = talib.CDLTHRUSTING(openn, high, low, close)
    
    return df