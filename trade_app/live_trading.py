import sqlalchemy
import pandas as pd
import matplotlib.pyplot as plt


from binance.client import Client
from binance import BinanceSocketManager
from sqlalchemy import create_engine
api_key = 'c7**8**JhG'
api_secret = 'VNb***Lf'
client = Client(api_key, api_secret)
engine = sqlalchemy.create_engine('sqlite:///BTCUSDTstream.db')
df = pd.read_sql('BTCUSDT', engine)
df.Price.plot()
plt.show()

def strategy(entry, lookback, qty, pen_position=False):
    while True:
        df = pd.read_sql('BTCUSDT', engine)
        lookbackperiod = df.iloc[-loockback:]
        cumret = (lookbackperiod.Price.pct_change() +1).cumprod() - 1
        if not open_position:
            if cumret[cumret.last_valid_index()] > entry:
                order = client.create_order(symbol='BTCUSDT', side='BUY', type='MARKET', quantity=qty)
                print(order)
                open_position = True
                break
                if open_position:
                    while True:
                        df = pd.read_sql('BTCUSDT', engine)
                        sincebuy = df.loc[df.Time > pd.to_datetime(order['transactTime'], unit='ms')]
                        if len(sincebuy) > 1:
                            sincebuyret = (sincebuy.Price.pct_change() +1).cumprod() - 1
                            last_entry = sincebuyret[sicncebuyret.last_valid_index()]
                            if last_entry > 0.0015 or last_entry < -0.0015:
                                order = client.create_order(symbol='BTCUSDT', side='SELL', type='MARKET', quantity=qty)
                                print(order)
                                break