import pandas as pd
import pandas_datareader as web
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

df = web.DataReader('AAPL', data_source='yahoo', start='2020-01-01', end='2020-05-10')

plt.figure(figsize=(12.2, 4.5))
plt.plot(df['Close'], label = 'Close')
plt.title('Close Price')
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Price USD', fontsize=18)
#plt.show()

OBV = []
OBV.append(0)

for i in range(1, len(df.Close)):
    if df.Close[i] > df.Close[i-1]:
        OBV.append(OBV[-1] + df.Volume[i])
    elif df.Close[i] < df.Close[i-1]:
        OBV.append(OBV[-1] - df.Volume[i])
    else:
        OBV.append(OBV[-1])

df['OBV'] = OBV
df['OBV_EMA'] = df['OBV'].ewm(span=20).mean()


plt.figure(figsize=(12.2, 4.5))
plt.plot(df['OBV'], label = 'OBV', color = 'orange')
plt.plot(df['OBV_EMA'], label = 'OBV_EMA', color = 'purple')
plt.title('OBV / OBV EMA Chart')
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Price USD', fontsize=18)
#plt.show()



def buy_sell(signal, col1, col2):
    sigPriceBuy = []
    sigPriceSell = []
    flag = -1

    for i in range (0, len(signal)):
        if signal[col1] [i] > signal[col2][i] and flag != 1:
            sigPriceBuy.append(signal['Close'][i])
            sigPriceSell.append(np.nan)
            flag = 1
        elif signal[col1] [i] < signal[col2] [i] and flag != 0:
            sigPriceSell.append(signal['Close'][i])
            sigPriceBuy.append(np.nan)
            flag = 0
        else:
            sigPriceSell.append(np.nan)
            sigPriceBuy.append(np.nan)
    return (sigPriceBuy, sigPriceSell)

x = buy_sell(df, 'OBV', 'OBV_EMA')
df['Buy_Signal_Price'] = x[0]
df['Sell_Signal_Price'] = x[1]


plt.figure(figsize=(12.2, 4.5))
plt.plot(df['Close'], label = 'Close', alpha = 0.35)
plt.scatter(df.index, df['Buy_Signal_Price'], label = 'Buy Signal', marker = '^', alpha=1, color = 'green')
plt.scatter(df.index, df['Sell_Signal_Price'], label = 'Sell Signal', marker = 'v', alpha=1, color = 'red')
plt.title('Buy and Sell Signals')
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Price USD', fontsize=18)
plt.legend(loc='upper left')
plt.show()