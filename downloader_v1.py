import ccxt
from ccxt.base.errors import RequestTimeout
import pandas as pd
from datetime import datetime
from datetime import timedelta
import time
import sys
import os
import shutil

binance = ccxt.binance()


def to_timestamp(dt):
    return binance.parse8601(dt.isoformat())


def download(symbol, start, end):
    '''
    Download all the transaction for a given symbol from the start date to the end date
    @param symbol: the symbol of the coin for which download the transactions
    @param start: the start date from which download the transaction
    @param end: the end date from which download the transaction
    '''

    records = []
    since = start
    ten_minutes = 60000 * 10

    print('Downloading {} from {} to {}'.format(symbol, binance.iso8601(start), binance.iso8601(end)))

    while since < end:
        #print('since: ' + binance.iso8601(since)) #uncomment this line of code for verbose download
        try:
            orders = binance.fetch_trades(symbol + '/BTC', since)
        except RequestTimeout:
            time.sleep(5)
            orders = binance.fetch_trades(symbol + '/BTC', since)

        if len(orders) > 0:

            latest_ts = orders[-1]['timestamp']
            if since != latest_ts:
                since = latest_ts
            else:
                since += ten_minutes

            for l in orders:
                print(l)
                records.append({
                    'symbol': l['symbol'],
                    'timestamp': l['timestamp'],
                    'datetime': l['datetime'],
                    'side': l['side'],
                    'price': l['price'],
                    'amount': l['amount'],
                    'btc_volume': float(l['price']) * float(l['amount']),
                })
        else:
            since += ten_minutes

    return pd.DataFrame.from_records(records)


def download_binance(days_before=7, days_after=7):
    '''
    Download all the transactions for all the pumps in binance in a given interval
    @param days_before: the number of days before the pump
    @param days_after: the number of days after the pump
    '''
    path = sys.argv[1]
    print("path:",path)
    df = pd.read_csv(path)
    binance_only = df[df['exchange'] == 'binance']
    print('数据总数：',binance_only.shape)

    number = 0
    for i, pump in binance_only.iterrows():
        symbol = pump['symbol']
        group = pump['group']
        date = pump['date'] + ' ' + pump['hour']
        pump_time = datetime.strptime(date, "%Y-%m-%d %H:%M")
        before = to_timestamp(pump_time - timedelta(days=days_before))
        after = to_timestamp(pump_time + timedelta(days=days_after))
        # to comment out
        src = 'data/{}_{}'.format(symbol, str(date).replace(':', '.') + '.csv')
        target = 'data/new_data/{}_{}_{}'.format(symbol,group, str(date).replace(':', '.') + '.csv')


        if os.path.exists(src) and os.path.exists(target) is False:
            print(symbol,' copy')
            shutil.copy(src,target)
            number += 1
            continue
        #
        df = download(symbol, before, after)
        # df.to_csv('data/new_data/{}_{}_{}'.format(symbol,group, str(date).replace(':', '.') + '.csv'), index=False)

    print('数据总数：',binance_only.shape[0])
    print("已下载数据总数：",number)

if __name__ == '__main__':
    download_binance(days_before=12, days_after=7)
