import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
import numpy as np

def classifier(time_freq,rolling_window,prefix='my'):

    computed_data = pd.read_csv('labeled_features/{}_features_{}_{}.csv'.format(prefix,time_freq,rolling_window), parse_dates=['date'])
    computed_data.fillna(1,inplace=True)
    features = [
                'std_rush_order',
                'avg_rush_order',
                'std_rush_order_price',
                'std_rush_order_amount',
                'std_trades',
                'std_volume',
                'avg_volume',
                'std_price',
                'avg_price',
                'avg_price_max',
                'avg_diff_min_max',
                'hour_sin',
                'hour_cos',
                'minute_sin',
                'minute_cos'
                ]

    X = computed_data[features]
    Y = computed_data['gt'].astype(int).values.ravel()
    # Y = [1] * len(X)

    # clf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=1)
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=100, learning_rate=0.5, random_state=0)

    cv_list = [5]
    processes = 8

    for n_fold in cv_list:
        print('Processing: {} folds - time freq {}'.format(n_fold, time_freq))
        y_pred = cross_val_predict(clf, X, Y.ravel(), cv=StratifiedKFold(n_splits=n_fold), n_jobs=processes)

        print('Recall: {}'.format(recall_score(Y, y_pred)))
        print('Precision: {}'.format(precision_score(Y, y_pred)))
        print('F1 score: {}'.format(f1_score(Y, y_pred)))
        # print("clf importance:{}".format(clf.feature_importances_))


if __name__ == '__main__':
    start = datetime.datetime.now()
    # classifier(time_freq='25S')
    classifier(time_freq='15S',rolling_window='240',prefix='my_new_sell')
    # classifier(time_freq='5S')
    print(datetime.datetime.now() - start)
