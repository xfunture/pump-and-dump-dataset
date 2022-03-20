import glob
import datetime
import os
import numpy as np
import pandas as pd

path = 'data/*.csv'

def std_rush_order_feature(df_buy, time_freq, rolling_freq):
    """
        compute std of rush order
        rush_order：orders with the same millionseconds time are combined
        构建std_rush_order 特征
        根据列time 进行groupby ,然后对数据重采样(降采样)，计算降采样后的25Sbtc_volume 的和
        这个函数复杂就复杂在你如何定义rush order，什么是rush order
    @param df_buy:
    @param time_freq:
    @param rolling_freq:
    @return:
    """
    # buy_volume = df_buy.groupby(pd.Grouper(freq=time_freq))['btc_volume'].sum()
    # buy_count = df_buy.groupby(pd.Grouper(freq=time_freq))['btc_volume'].count()
    # buy_volume.drop(buy_count[buy_count == 0].index, inplace=True)
    # rolling_df = buy_volume.rolling(window=rolling_freq).std()
    # results = rolling_df.pct_change()
    # results.reset_index().set_index('time')

    df_buy = df_buy.groupby(df_buy.index).count()
    df_buy[df_buy == 1] = 0
    df_buy[df_buy > 1] = 1
    buy_volume = df_buy.groupby(pd.Grouper(freq=time_freq))['btc_volume'].sum()
    buy_count = df_buy.groupby(pd.Grouper(freq=time_freq))['btc_volume'].count()
    buy_volume.drop(buy_volume[buy_count == 0].index, inplace=True)
    buy_volume.dropna(inplace=True)
    rolling_diff = buy_volume.rolling(window=rolling_freq).std()
    results = rolling_diff.pct_change()

    return results

def std_rush_order_feature_sell(df_buy, time_freq, rolling_freq):
    """
        compute std of rush order
        rush_order：orders with the same millionseconds time are combined
        构建std_rush_order 特征
        根据列time 进行groupby ,然后对数据重采样(降采样)，计算降采样后的25Sbtc_volume 的和
        这个函数复杂就复杂在你如何定义rush order，什么是rush order
    @param df_buy:
    @param time_freq:
    @param rolling_freq:
    @return:
    """
    # buy_volume = df_buy.groupby(pd.Grouper(freq=time_freq))['btc_volume'].sum()
    # buy_count = df_buy.groupby(pd.Grouper(freq=time_freq))['btc_volume'].count()
    # buy_volume.drop(buy_count[buy_count == 0].index, inplace=True)
    # rolling_df = buy_volume.rolling(window=rolling_freq).std()
    # results = rolling_df.pct_change()
    # results.reset_index().set_index('time')

    df_buy = df_buy.groupby(df_buy.index).count()
    df_buy[df_buy == 1] = 0
    df_buy[df_buy > 1] = 1
    buy_volume = df_buy.groupby(pd.Grouper(freq=time_freq))['btc_volume'].sum()
    buy_count = df_buy.groupby(pd.Grouper(freq=time_freq))['btc_volume'].count()
    buy_volume.drop(buy_volume[buy_count == 0].index, inplace=True)
    buy_volume.dropna(inplace=True)
    rolling_diff = buy_volume.rolling(window=rolling_freq).std()
    results = rolling_diff.pct_change()

    return results


def avg_rush_order_feature(df_buy, time_freq, rolling_freq):
    """
        根据buy order 计算rush Order ,然后在对数据进行重采样，最后滚动计算交易量(volume)的的平均值
    @param df_buy:
    @param time_freq:
    @param rolling_freq:
    @return:
    """
    # buy_volume = df_buy.groupby(pd.Grouper(freq=time_freq))['btc_volume'].sum()
    # buy_count = df_buy.groupby(pd.Grouper(freq=time_freq))['btc_volume'].count()
    # buy_volume.drop(buy_volume[buy_count == 0].index, inplace=True)
    # buy_volume.dropna(inplace=True)
    # rolling_diff = buy_volume.rolling(window=rolling_freq).mean()
    # results = rolling_diff.pct_change()
    # results.reset_index().set_index('time')

    df_buy = df_buy.groupby(df_buy.index).count()
    df_buy[df_buy == 1] = 0
    df_buy[df_buy > 1] = 1
    buy_volume = df_buy.groupby(pd.Grouper(freq=time_freq))['btc_volume'].sum()
    buy_count = df_buy.groupby(pd.Grouper(freq=time_freq))['btc_volume'].count()
    buy_volume.drop(buy_volume[buy_count == 0].index, inplace=True)
    buy_volume.dropna(inplace=True)
    rolling_diff = buy_volume.rolling(window=rolling_freq).std()
    results = rolling_diff.pct_change()
    return results

def avg_rush_order_feature_sell(df_buy, time_freq, rolling_freq):
    """
        根据buy order 计算rush Order ,然后在对数据进行重采样，最后滚动计算交易量(volume)的的平均值
    @param df_buy:
    @param time_freq:
    @param rolling_freq:
    @return:
    """
    # buy_volume = df_buy.groupby(pd.Grouper(freq=time_freq))['btc_volume'].sum()
    # buy_count = df_buy.groupby(pd.Grouper(freq=time_freq))['btc_volume'].count()
    # buy_volume.drop(buy_volume[buy_count == 0].index, inplace=True)
    # buy_volume.dropna(inplace=True)
    # rolling_diff = buy_volume.rolling(window=rolling_freq).mean()
    # results = rolling_diff.pct_change()
    # results.reset_index().set_index('time')

    df_buy = df_buy.groupby(df_buy.index).count()
    df_buy[df_buy == 1] = 0
    df_buy[df_buy > 1] = 1
    buy_volume = df_buy.groupby(pd.Grouper(freq=time_freq))['btc_volume'].sum()
    buy_count = df_buy.groupby(pd.Grouper(freq=time_freq))['btc_volume'].count()
    buy_volume.drop(buy_volume[buy_count == 0].index, inplace=True)
    buy_volume.dropna(inplace=True)
    rolling_diff = buy_volume.rolling(window=rolling_freq).std()
    results = rolling_diff.pct_change()
    return results


def std_rush_order_price(df_buy,time_freq,rolling_freq):
    """
        新特征，提升精确度
    """
    df_buy = df_buy.groupby(df_buy.index).count()
    df_buy[df_buy == 1] = 0
    df_buy[df_buy > 1] = 1
    buy_volume = df_buy.groupby(pd.Grouper(freq=time_freq))['price'].mean()
    buy_count = df_buy.groupby(pd.Grouper(freq=time_freq))['price'].count()
    buy_volume.drop(buy_volume[buy_count == 0].index, inplace=True)
    buy_volume.dropna(inplace=True)
    rolling_diff = buy_volume.rolling(window=rolling_freq).std()
    results = rolling_diff.pct_change()
    return results

def std_rush_order_amount(df_buy,time_freq,rolling_freq):
    """
        新特征，对模型性能没有提升
    """
    df_buy = df_buy.groupby(df_buy.index).count()
    df_buy[df_buy == 1] = 0
    df_buy[df_buy > 1] = 1
    buy_volume = df_buy.groupby(pd.Grouper(freq=time_freq))['amount'].mean()
    buy_count = df_buy.groupby(pd.Grouper(freq=time_freq))['amount'].count()
    buy_volume.drop(buy_volume[buy_count == 0].index, inplace=True)
    buy_volume.dropna(inplace=True)
    rolling_diff = buy_volume.rolling(window=rolling_freq).std()
    results = rolling_diff.pct_change()
    return results

def std_trades_feature(df_buy_rolling, rolling_freq):
    """
        计算重采样后的交易次数，pandas count 计算非NAN的数量
    @param df_buy_rolling:
    @param rolling_freq: 滚动窗口大小
    @return:
    """
    buy_volume = df_buy_rolling['price'].count()
    buy_volume.drop(buy_volume[buy_volume == 0].index, inplace=True)
    buy_volume.dropna(inplace=True)
    rolling_diff = buy_volume.rolling(window=rolling_freq).std()
    results = rolling_diff.pct_change()
    return results


def std_volume_feature(df_buy_rolling, rolling_freq):
    """
        计算重采样后交易量的标准差
    @param df_buy_rolling:
    @param rolling_freq:
    @return:
    """
    buy_volume = df_buy_rolling['btc_volume'].sum()
    buy_volume.drop(buy_volume[buy_volume == 0].index, inplace=True)
    buy_volume.dropna(inplace=True)
    rolling_diff = buy_volume.rolling(window=rolling_freq).std()
    results = rolling_diff.pct_change()
    return results


def avg_volume_feature(df_buy_rolling, rolling_freq):
    """
    计算重采样后交易量的平均值，然后再在交易量平均值的基础上再计算百分比变化
    @param df_buy_rolling:
    @param rolling_freq:
    @return:
    """
    buy_volume = df_buy_rolling['btc_volume'].sum()
    buy_volume.drop(buy_volume[buy_volume == 0].index, inplace=True)
    buy_volume.dropna(inplace=True)
    rolling_diff = buy_volume.rolling(window=rolling_freq).mean()
    results = rolling_diff.pct_change()
    return results


def std_price_feature(df_buy_rolling, rolling_freq):
    """
    先计算重采样后价格的标准差，然后再在价格标准差的基础上再计算百分比变化
    @param df_buy_rolling:
    @param rolling_freq:
    @return:
    """
    buy_volume = df_buy_rolling['price'].mean()
    buy_volume.dropna(inplace=True)
    rolling_diff = buy_volume.rolling(window=rolling_freq).std()
    results = rolling_diff.pct_change()
    return results


def avg_price_feature(df_buy_rolling):
    """
    先计算重采样后价格的平均值，然后再在价格平均值的基础上再计算百分比变化
    @param df_buy_rolling:
    @return:
    """
    buy_volume = df_buy_rolling['price'].mean()
    buy_volume.dropna(inplace=True)
    rolling_diff = buy_volume.rolling(window=10).mean()
    results = rolling_diff.pct_change()
    return results

def avg_diff_min_max(df_buy_grouped,rolling_freq):
    """
        计算最大值和最小值的移动差的平均值
        添加该特征后，模型性能没有提高
    @param df_buy:
    @return:
    """
    results = df_buy_grouped['price'].max() - df_buy_grouped['price'].min()
    results.dropna(inplace=True)
    rolling_diff = results.rolling(window=rolling_freq).mean()
    return rolling_diff.pct_change()

def avg_price_max(df_buy_rolling):
    """
        先计算重采样后价格的最大值，然后再在价格最大值的基础上计算平均值，最后算百分比变化
    @param df_buy_rolling:
    @return:
    """
    buy_volume = df_buy_rolling['price'].max()
    buy_volume.dropna(inplace=True)
    rolling_diff = buy_volume.rolling(window=10).mean()
    results = rolling_diff.pct_change()
    return results


def chunks_time(df_buy_rolling):
    """
        返回chunk 的时间索引
    @param df_buy_rolling:
    @return:
    """
    # compute any kind of aggregation
    buy_volume = df_buy_rolling['price'].max()
    buy_volume.dropna(inplace=True)
    #the index contains time info
    return buy_volume.index


def build_features(file, coin, time_freq, rolling_freq, index):
    """
        构建特征向量
    @param file: 文件
    @param coin: 币种
    @param time_freq:重采样频率,25S,15S,5S
    @param rolling_freq:滚动窗口大小
    @param index:  ？？？
    @return:
    """
    df = pd.read_csv(file)
    df["time"] = pd.to_datetime(df['timestamp'].astype(np.int64), unit='ms')
    df = df.reset_index().set_index('time')

    df_buy = df[df['side'] == 'buy']
    df_sell = df[df['side'] == 'sell']


    df_buy_grouped = df_buy.groupby(pd.Grouper(freq=time_freq))

    #获取groupby 后的索引，将时间序列按照25s的周期进行重采样
    #该函数只是返回索引
    date = chunks_time(df_buy_grouped)

    results_df = pd.DataFrame(
        {'date': date,
         'pump_index': index,
         'std_rush_order': std_rush_order_feature(df_buy, time_freq, rolling_freq).values,
         'std_rush_order_sell':std_rush_order_feature_sell(df_sell,time_freq,rolling_freq).values,
         'avg_rush_order': avg_rush_order_feature(df_buy, time_freq, rolling_freq).values,
         'avg_rush_order_sell': avg_rush_order_feature_sell(df_sell,time_freq,rolling_freq).values,
         'std_rush_order_price':std_rush_order_price(df_buy,time_freq,rolling_freq).values,  #新添加特征，有用
         'std_rush_order_amount':std_rush_order_amount(df_buy,time_freq,rolling_freq).values,#新添加特征,有用
         'std_trades': std_trades_feature(df_buy_grouped, rolling_freq).values,
         'std_volume': std_volume_feature(df_buy_grouped, rolling_freq).values,
         'avg_volume': avg_volume_feature(df_buy_grouped, rolling_freq).values,
         'std_price': std_price_feature(df_buy_grouped, rolling_freq).values,
         'avg_price': avg_price_feature(df_buy_grouped),
         'avg_price_max': avg_price_max(df_buy_grouped).values,
         'avg_diff_min_max':avg_diff_min_max(df_buy_grouped,rolling_freq).values,           #新添加特征
         'hour_sin': np.sin(2 * np.pi * date.hour/23),
         'hour_cos': np.cos(2 * np.pi * date.hour/23),
         'minute_sin': np.sin(2 * np.pi * date.minute / 59),
         'minute_cos': np.cos(2 * np.pi * date.minute / 59),
         })

    results_df['symbol'] = coin
    results_df['gt'] = 0
    results_df.replace([np.inf,-np.inf],np.nan,inplace=True)
    return results_df.dropna()


def build_features_multi(time_freq, rolling_freq,prefix="my"):
    """
    @param time_freq:重采样频率，25S,15S,5S
    @param rolling_freq:滚动窗口大小
    @return:
    """
    files = glob.glob(path)

    all_results_df = pd.DataFrame()
    count = 0
    pumps = pd.read_csv('pump_telegram.csv')
    pumps = pumps[pumps['exchange'] == 'binance']

    for index,f in enumerate(files):
        print(index,f)
        coin_date, time = os.path.basename(f[:f.rfind('.')]).split(' ')
        coin, date = coin_date.split('_')

        skip_pump = len(pumps[(pumps['symbol'] == coin) & (pumps['date'] == date) & (pumps['hour'] == time.replace('.', ':'))]) == 0
        if skip_pump:
            continue

        results_df = build_features(f, coin, time_freq, rolling_freq, count)

        date_datetime = datetime.datetime.strptime(date + ' ' + time, '%Y-%m-%d %H.%M')

        # We consider 24 hours before and 24 hours after the pump
        results_df = results_df[(results_df['date'] >= date_datetime - datetime.timedelta(hours=24)) & (results_df['date'] <= date_datetime + datetime.timedelta(hours=24))]

        all_results_df = pd.concat([all_results_df, results_df])
        count += 1
    all_results_df.fillna(value=0,inplace=True)
    all_results_df.to_csv('features/{}_features_{}_{}.csv'.format(prefix,time_freq,rolling_freq), index=False, float_format='%.3f')



def merge_features_and_label(time_freq,rolling_freq,prefix='my'):
    """
        合并特征文件和标签文件
    @return:
    """
    # 读取特征文件
    features_path = "features/{}_features_{}_{}.csv".format(prefix,time_freq,rolling_freq)
    df_features = pd.read_csv(features_path)
    # df_features.set_index(pd.to_datetime(df_features['date']),inplace=True)
    print("features shape", df_features.shape)

    # 读取25S 特征的标签文件
    label_path = "labeled_features/label_{}.csv".format(time_freq)
    df_label = pd.read_csv(label_path)
    df_label = df_label.loc[:, ['date', 'symbol', 'gt']]
    df_label['gt'] = df_label['gt'].astype('int')
    print("label file gt == 1 shape:", df_label.shape)
    # df_label_output = df_label[df_label['gt']==1]
    # df_label.set_index(pd.to_datetime(df_label['date']),inplace=True)
    # display(df_label.loc['2018-12-30 17:00:00'])

    # 合并特征文件和标签文件
    output_path = 'labeled_features/{}_features_{}_{}.csv'.format(prefix,time_freq,rolling_freq)
    new_df = pd.merge(left=df_features, right=df_label, on=['symbol', 'date'], how='left')

    new_df['gt'] = new_df['gt_y']
    new_df.drop(columns=['gt_x', 'gt_y'], inplace=True)
    new_df['gt'].fillna(inplace=True, value=0)
    new_df.to_csv(output_path, index=False)
    print("label == 1 shape:", new_df[new_df['gt'] == 1].shape)

def compute_features():
    # build_features_multi(time_freq='15S', rolling_freq=144,prefix="my_new")     #设置rolling window 的大小
    # merge_features_and_label(time_freq='15S',rolling_freq=144,prefix="my_new")

    build_features_multi(time_freq='15S', rolling_freq=240,prefix="my_new_buy_and_sell")     #设置rolling window 的大小
    merge_features_and_label(time_freq='15S',rolling_freq=240,prefix="my_new_buy_and_sell")

    # build_features_multi(time_freq='15S', rolling_freq=720,prefix="my_new")      #设置rolling window 的大小
    # merge_features_and_label(time_freq='15S',rolling_freq=240,prefix="my_new")

if __name__ == '__main__':
    start = datetime.datetime.now()
    compute_features()
    print(datetime.datetime.now() - start)
