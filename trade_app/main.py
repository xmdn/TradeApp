import pandas as pd
import asyncio
import sqlalchemy
from binance import BinanceSocketManager
from binance.client import Client
import time


async def main():
    api_key = 'c7***1JhG'
    api_secret = 'VNbr*****fSALf'
    client = Client(api_key, api_secret)
    bsm = BinanceSocketManager(client)
    socket = bsm.trade_socket('BTCUSDT')


    while True:

        await socket.__aenter__()
        msg = await socket.recv()

        def createframe(msg):
            df = pd.DataFrame([msg])
            df = df.loc[:,['s','E','p']]
            df.columns = ['symbol', 'Time', 'Price']
            df.Price = df.Price.astype(float)
            df.Time = pd.to_datetime(df.Time, unit='ms')
            return df

        frame = createframe(msg)
        engine = sqlalchemy.create_engine('sqlite:///BTCUSDTstream.db')
        frame.to_sql('BTCUSDT', con=engine, if_exists='append', index=False)
        print(frame)
        time.sleep(1)

if __name__ == '__main__':
    asyncio.run(main())

