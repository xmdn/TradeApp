import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
plt.style.use('fivethirtyeight')

df = web.DataReader('AAPL', data_source='yahoo', start='2020-01-01', end='2020-05-10')

plt.figure(figsize=(12.2, 4.5))
plt.plot(df['Close'], label = 'Close')
plt.xticks(rotation=45)
plt.title('Close Price')
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Price USD', fontsize=18)
plt.show()

ShortEMA = df.Close.ewm(span=12, adjust=False).mean()
LongEMA = df.Close.ewm(span=26, adjust=False).mean()
MACD = ShortEMA - LongEMA
signal = MACD.ewm(span=9, adjust=False).mean()

plt.figure(figsize=(12.2, 4.5))
plt.plot(df.index, MACD, label = 'AAPL MACD', color='red')
plt.plot(df.index, signal, label = 'Signal Line', color='blue')
plt.xticks(rotation=45)
plt.legend(loc='upper left')
plt.show()

df['MACD'] = MACD
df['Signal Line'] = signal


def buy_sell(signal):
    Buy = []
    Sell = []
    flag = -1

    for i in range (0, len(signal)):
        if signal['MACD'][i] > signal['Signal Line'][i]:
            Sell.append(np.nan)
            if flag != 1:
                Buy.append(signal['Close'][i])
                flag = 1
            else:
                Buy.append(np.nan)
        elif signal['MACD'] [i] < signal['Signal Line'] [i]:
            Buy.append(np.nan)
            if flag != 0:
                Sell.append(signal['Close'][i])
                flag = 0
            else:
                Sell.append(np.nan)
        else:
            Buy.append(np.nan)
            Sell.append(np.nan)
    return (Buy, Sell)

a = buy_sell(df)
df['Buy_Signal_Price'] = a[0]
df['Sell_Signal_Price'] = a[1]

plt.figure(figsize=(12.2, 4.5))
plt.scatter(df.index, df['Buy_Signal_Price'], label = 'Buy', marker='^', color='green', alpha=1)
plt.scatter(df.index, df['Sell_Signal_Price'], label = 'Sell', marker='^', color='red', alpha=1)
plt.plot(df['Close'], label='Close Price', alpha=0.35)
plt.title('Close Price Buy & Sell Signals')
plt.xticks(rotation=45)
plt.xlabel('Data')
plt.xlabel('Close Price USD ($)')
plt.legend(loc='upper left')
plt.show()