# Multistep_LSTM_on_XRPUSDT
Trains stacked bidirectional multi-step LSTM on XRPUSDT OHLC data to predict high volatility buy/sell zones. Then uses this signals to enter positions via Metatrader5
python API.

## 01_download_monthly_data.py
Gets all available XRPUSDT 1H OHLC data from Binance using Binance API starting from 2018 to 2022

## 02_preprocess_data.py
Creates target classes. Class 1 is the point where in the next 12 hour price increases more than %5 and Class 2 is the point where in the next 12 hour price decreases
more than %5. All other instances/points are labeled as class 0. This script also applies **feature_engineering.py** on OHLC and volume data to create new features.

## feature_engineering.py
Creates multiple indicator (BB, MACD, MFI, ADX etc.) and candlestick pattern (hammer, engulfing, long leg doji, hanging, 3 white soldiers etc.) features to extract
more information from OHLC and volume data.

## 03_reshape.py
Reshapes time series data into a shape of (None, MULTI_STEP_SIZE, NUMBER_OF_FEATURES) numpy array to feed them into multistep LSTM. Uses garbage collection to 
increase memory efficiency.

## 04_model.py
Gets class_weights automatically from training dataset to feed them later into model.fit(..class_weight = class_weights) to train efficiently on unbalanced data.
Then trains Stacked Bidirectional Multi-step LSTM on training data and saves the model.

## 05_analyze_predictions.py
Loads saved LSTM model and analyzes its predictions via plot and also applies a simple backtesting to check profitability on different take profit and stop loss levels.

![plot](https://user-images.githubusercontent.com/103101771/172843573-44910245-a0ea-4ce8-b7af-497adc037f64.png)

Here white-red dots represent closing price of XRPUSDT. More the color is darker, more the volatility in next 12 hours. Purple points over prices represent short signals
and blue dots below price represent long signals.

## 06_mt5_api.py
Connects to metatrader5 account via metatrader5 python api and enters long/short positions according to predictions of the model. I did not finish this part but still there is a template that you can use.





