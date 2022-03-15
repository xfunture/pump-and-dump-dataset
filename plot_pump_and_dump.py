import pandas as pd
import numpy as np
from datetime import datetime
import os
import ccxt
from datetime import timedelta
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")
binance = ccxt.binance()

def to_timestamp(dt):
    return binance.parse8601(dt.isoformat())


def plot_data():
    path = 'pump_telegram.csv'
    print("path:", path)
    data = pd.read_csv(path)
    binance_only = data[data['exchange'] == 'binance'][250:300]
    print('数据总数：', binance_only.shape[0])

    number = 0
    for i, pump in binance_only.iterrows():
        symbol = pump['symbol']
        group = pump['group']
        date = pump['date'] + ' ' + pump['hour']
        pump_time = datetime.strptime(date, "%Y-%m-%d %H:%M")
        before_time = pump_time - timedelta(minutes=360)
        after_time = pump_time + timedelta(minutes=2)
        file_path = 'data/{}_{}'.format(symbol, str(date).replace(':', '.') + '.csv')
        title = '{}_{}_{}'.format(symbol,group,pump_time)
        if os.path.exists(file_path) is False:
            print("not found:",symbol,group)
            continue
        df = pd.read_csv(file_path)
        print("df shape: {} pump_time:{}".format(df.shape,pump_time),before_time,after_time)
        df["time"] = pd.to_datetime(df['timestamp'].astype(np.int64), unit='ms')
        df = df.reset_index().set_index('time')
        df_buy = df[df['side'] == 'buy']
        pump_price_data = df.loc[before_time:after_time, ['price']]
        pump_volume_data = df.loc[before_time:after_time,['btc_volume']]
        pump_amount_data = df.loc[before_time:after_time,['amount']]
        print(pump_price_data.head())
        fig = plt.figure(figsize=(20,5))
        plt.subplot(311)
        plt.plot(pump_price_data)
        plt.subplot(312)
        plt.plot(pump_volume_data)
        plt.subplot(313)
        plt.plot(pump_amount_data)
        plt.title(title)
        plt.show()




def main():
    plot_data()





if __name__ == "__main__":
    main()