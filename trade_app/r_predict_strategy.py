import sqlalchemy
import pandas as pd
from binance import BinanceSocketManager
import matplotlib.pyplot as plt
import yfinance as yf

google = yf.Ticker('GOOG')

df = google.history(period='1d', interval="1m")
print(df.head())


X = df.index.values
y = df['Low'].values
# The split point is the 10% of the dataframe length
offset = int(0.10 * len(df))
X_train = X[:-offset]
y_train = y[:-offset]
X_test = X[-offset:]
y_test = y[-offset:]

plt.plot(range(0, len(y_train)), y_train, label='Train')
plt.plot(range(len(y_train), len(y)), y_test, label='Test')
plt.legend()
plt.show()

from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(y_train, order=(5, 0, 1)).fit()
forecast = model.forecast(steps=1)[0]

print(f'Real data for time 0: {y_train[len(y_train) - 1]}')
print(f'Real data for time 1: {y_test[0]}')
print(f'Pred data for time 1: {forecast}')
