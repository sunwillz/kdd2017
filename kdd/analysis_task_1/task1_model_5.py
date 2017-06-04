# -*- coding: utf-8 -*-
"""
BP神经网络预测模型
设Vi(t)为某一路段i在时段t的交通流量,Vi(t-1)为路段i在时段t-1的交通流量,
采用过去几个时段(20分钟)的流量对未来某个时段的流量进行预测
X：Vi(t-1),Vi(t-2),Vi(t-3),Vi(t-4)......
Y:Vi(t)
划分训练集：2016.07.19-2016.10.10(应当考虑去掉特殊日期，如国庆假期和中秋假期)
验证集：2016.10.11-2016.10.17
测试集：2016.10.18-2016.10.24
"""

import os
import datetime
from os import path
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn import neighbors
from kdd.metrics import task1_eva_metrics

PROJECT_PATH = os.path.dirname(os.path.abspath(__name__))
dir_path = path.join(PROJECT_PATH, "datasets/dataSets/")
train_data_path = path.join(dir_path, "training/")
test_data_path = path.join(dir_path, "testing_phase1/")

training_file = 'trajectories(table 5)_training.csv'
testing_file = 'trajectories(table 5)_test1.csv'


def read_training_data(filename):
    filepath = train_data_path + filename
    return pd.read_csv(filepath, header=0)


def read_testing_data(filename):
    filepath = test_data_path + filename
    return pd.read_csv(filepath, header=0)


def trainging_data():
    data_df = read_training_data(training_file)
    start_date = '2016-07-19 00:00:00'
    end_date = '2016-10-10 23:59:59'
    data_df = data_df[(data_df.starting_time >= start_date) & (data_df.starting_time <= end_date)]
    return data_df


def validation_data():
    data_df = read_training_data(training_file)
    start_date = '2016-10-11 00:00:00'
    end_date = '2016-10-17 23:59:59'
    data_df = data_df[(data_df.starting_time >= start_date) & (data_df.starting_time <= end_date)]
    return data_df


def testing_data():
    data_df = read_testing_data(testing_file)
    return data_df


def write_to_file(data, filename):
    """
    将结果写入文件
    :param filename:
    :param data:
    :return:
    """
    data.to_csv(PROJECT_PATH + "/" + filename, index=False)


def get_avg_time(flag=True):
    """
    :param flag:前两小时为True,后两小时为False
    :return:
    """
    data = read_training_data('trajectories(table 5)_training.csv')
    if flag:
        time_inteval = [
            ['06:00:00', '06:20:00'], ['06:20:00', '06:40:00'], ['06:40:00', '07:00:00'],
            ['07:00:00', '07:20:00'], ['07:20:00', '07:40:00'], ['07:40:00', '08:00:00'],
            ['15:00:00', '15:20:00'], ['15:20:00', '15:40:00'], ['15:40:00', '16:00:00'],
            ['16:00:00', '16:20:00'], ['16:20:00', '16:40:00'], ['16:40:00', '17:00:00']]
    else:
        time_inteval = [
            ['08:00:00', '08:20:00'], ['08:20:00', '08:40:00'], ['08:40:00', '09:00:00'],
            ['09:00:00', '09:20:00'], ['09:20:00', '09:40:00'], ['09:40:00', '10:00:00'],
            ['17:00:00', '17:20:00'], ['17:20:00', '17:40:00'], ['17:40:00', '18:00:00'],
            ['18:00:00', '18:20:00'], ['18:20:00', '18:40:00'], ['18:40:00', '19:00:00']]
    roads = [['A', 2], ['A', 3], ['B', 1], ['B', 3], ['C', 1], ['C', 3]]
    trajectories = [
        [110, 123, 107, 108, 120, 117],
        [123, 107, 108, 119, 114, 118, 122],
        [105, 100, 111, 103, 116, 101, 121, 106, 113],
        [105, 100, 111, 103, 122],
        [115, 102, 109, 104, 112, 111, 103, 116, 101, 121, 106, 113],
        [115, 102, 109, 104, 112, 111, 103, 122]
    ]
    link_time = {}
    for index, row in data.iterrows():
        trajectory = row['travel_seq']
        links = trajectory.split(';')
        start_time_date = links[0].split('#')[1]
        for link in links:
            _array = link.split('#')
            link_id = _array[0]
            start_time_time = _array[1].split(' ')[1]
            travel_time = _array[2]
            if link_id not in link_time.keys():
                link_time[link_id] = {}
                for window in range(1, 13):
                    link_time[link_id]['travel_time' + str(window)] = []
            for index, item in enumerate(time_inteval):
                if item[0] <= start_time_time <= item[1]:
                    link_time[link_id]['travel_time' + str(index + 1)].append(float(travel_time))

    for link_id, time_dic in link_time.iteritems():
        for window, time_list in time_dic.iteritems():
            _list = link_time[link_id][window]
            _len = len(_list)
            _list.sort()
            _list = _list[int(_len * 0.1):int(_len * 0.9)]
            _list_mean = float(sum(_list)) / len(_list)
            link_time[link_id][window] = _list_mean
    traj = {}
    for idx, trajectory in enumerate(trajectories):
        key = str(roads[idx][0]) + '->' + str(roads[idx][1])
        if key not in traj:
            traj[key] = {}
        for time_window in range(1, 13):
            travel_time = 0
            for link_id in trajectory:
                travel_time += link_time[str(link_id)]['travel_time' + str(time_window)]
            traj[key]['travel_time' + str(time_window)] = travel_time
    return traj


def set_missing_data(df, flag):
    """
    用trajectory的平均时间代替缺失值
    :param flag:
    :param df:
    :return:
    """
    traj = get_avg_time(flag=flag)
    col_names = ['travel_time' + str(x) for x in range(1, 13)]
    result = pd.DataFrame(columns=df.columns)
    for index, row in df.iterrows():
        inte_id = row['intersection_id']
        to_id = row['tollgate_id']
        for col in col_names:
            if row[col] != row[col]:  # 为缺失值
                row[col] = traj[str(inte_id) + '->' + str(int(to_id))][col]
        result = result.append(pd.DataFrame(np.array([row]), columns=result.columns), ignore_index=True)
    return result


def feature_extra(train_df, start_date="2016-07-18", end_date="2016-10-10"):
    time_window = [['06:00:00', '06:20:00'], ['06:20:00', '06:40:00'], ['06:40:00', '07:00:00'],
                   ['07:00:00', '07:20:00'], ['07:20:00', '07:40:00'], ['07:40:00', '08:00:00'],
                   ['15:00:00', '15:20:00'], ['15:20:00', '15:40:00'], ['15:40:00', '16:00:00'],
                   ['16:00:00', '16:20:00'], ['16:20:00', '16:40:00'], ['16:40:00', '17:00:00']]
    date1 = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    date2 = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    i = datetime.timedelta(days=1)
    columns_names = ['intersection_id', 'tollgate_id', 'date']
    data_df = pd.DataFrame(columns=columns_names)
    while i <= (date2 - date1):
        cur_date = (date1 + i).strftime('%Y-%m-%d')
        for index, inteval in enumerate(time_window):
            df = train_df[(train_df.starting_time >= cur_date + " " + inteval[0]) & (
                train_df.starting_time <= cur_date + " " + inteval[1])]
            df = df.groupby(['intersection_id', 'tollgate_id']).mean().reset_index()
            df['date'] = cur_date
            df['travel_time' + str(index + 1)] = df.travel_time
            new_data_df = df[['intersection_id', 'tollgate_id', 'date', 'travel_time' + str(index + 1)]]
            data_df = data_df.append(new_data_df, ignore_index=True)
        i += datetime.timedelta(days=1)
    data_df = data_df.groupby(['intersection_id', 'tollgate_id', 'date']).sum().reset_index()

    data_df = set_missing_data(data_df, flag=True)
    for idx in range(12, 0, -1):
        col = data_df.pop('travel_time' + str(idx))
        data_df.insert(0, 'travel_time' + str(idx), col)
    return data_df


def label_extra(train_df, start_date="2016-07-18", end_date="2016-10-10"):
    time_window = [['08:00:00', '08:20:00'], ['08:20:00', '08:40:00'], ['08:40:00', '09:00:00'],
                   ['09:00:00', '09:20:00'], ['09:20:00', '09:40:00'], ['09:40:00', '10:00:00'],
                   ['17:00:00', '17:20:00'], ['17:20:00', '17:40:00'], ['17:40:00', '18:00:00'],
                   ['18:00:00', '18:20:00'], ['18:20:00', '18:40:00'], ['18:40:00', '19:00:00']]
    date1 = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    date2 = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    i = datetime.timedelta(days=1)
    data_df = pd.DataFrame(columns=train_df.columns)
    while i <= (date2 - date1):
        cur_date = (date1 + i).strftime('%Y-%m-%d')
        for index, inteval in enumerate(time_window):
            df = train_df[(train_df.starting_time >= cur_date + " " + inteval[0]) & (
                train_df.starting_time <= cur_date + " " + inteval[1])]
            df = df.groupby(['intersection_id', 'tollgate_id']).mean().reset_index()
            df['date'] = cur_date
            df['travel_time' + str(index + 1)] = df.travel_time
            new_data_df = df[['intersection_id', 'tollgate_id', 'date', 'travel_time' + str(index + 1)]]
            data_df = data_df.append(new_data_df, ignore_index=True)
        i += datetime.timedelta(days=1)
    data_df = data_df.groupby(['intersection_id', 'tollgate_id', 'date']).sum().reset_index()
    data_df = set_missing_data(data_df, flag=False)
    for idx in range(12, 0, -1):
        col = data_df.pop('travel_time' + str(idx))
        data_df.insert(0, 'travel_time' + str(idx), col)
    return data_df


def data_process(X_train_df, y_train_df):
    """
    从训练数据中提取具有相同日期的数据
    :param X_train_df:
    :param y_train_df:
    :return:
    """
    arrX = np.array(X_train_df['date'])
    arrY = np.array(y_train_df['date'])
    arr = np.array(list(set(arrX).intersection(set(arrY))))
    X_train_df = X_train_df[X_train_df.date.isin(arr)]
    y_train_df = y_train_df[y_train_df.date.isin(arr)]
    return X_train_df, y_train_df


def to_submit_format_df(df):
    """
    :param flag:
    :param df:
    :param flag1: 是否上午，上午为true,下午为False
    :return:
    """
    time_inteval = [
        ['08:00:00', '08:20:00'], ['08:20:00', '08:40:00'], ['08:40:00', '09:00:00'],
        ['09:00:00', '09:20:00'], ['09:20:00', '09:40:00'], ['09:40:00', '10:00:00'],
        ['17:00:00', '17:20:00'], ['17:20:00', '17:40:00'], ['17:40:00', '18:00:00'],
        ['18:00:00', '18:20:00'], ['18:20:00', '18:40:00'], ['18:40:00', '19:00:00']
    ]
    result_df = []
    for index, row in df.iterrows():
        _intersection_id = row['intersection_id']
        _tollgate_id = row['tollgate_id']
        _date = row['date']
        for col_name in df.columns:
            if 'travel_time' in col_name:
                tp_list = []
                tp_list.append(_intersection_id)
                tp_list.append(str(int(_tollgate_id)))
                idx = int(re.findall(r'\d+', col_name)[0])
                time_window = "[" + _date + " " + time_inteval[idx - 1][0] + "," + _date + " " + time_inteval[idx - 1][
                    1] + ")"
                tp_list.append(time_window)
                tp_list.append(row[col_name])
                result_df.append(tp_list)

    return pd.DataFrame(result_df, columns=['intersection_id', 'tollgate_id', 'time_window', 'avg_travel_time'])


def local_test():
    predict_df = pd.DataFrame(columns=['intersection_id', 'tollgate_id', 'time_window', 'avg_travel_time'])
    roads = [['A', 2], ['A', 3], ['B', 1], ['B', 3], ['C', 1], ['C', 3]]

    for trajectory in roads:
        intersection_id = trajectory[0]
        tollgate_id = trajectory[1]
        train_df = read_training_data(training_file)
        train_df = train_df[(train_df.intersection_id == intersection_id) & (train_df.tollgate_id == tollgate_id)]
        X_train_df = feature_extra(train_df)
        y_train_df = label_extra(train_df)
        X_test_df = feature_extra(train_df, start_date='2016-10-10', end_date='2016-10-18')
        X_train_df, y_train_df = data_process(X_train_df, y_train_df)
        ret = []
        for flag in [True, False]:
            if flag:
                X_train = X_train_df.iloc[:, 0:6]
                X_train = X_train.astype(float)
                X_test = X_test_df.iloc[:, 0:6]
                X_test = X_test.astype(float)
                for i in range(6):
                    y_train = y_train_df.iloc[:, i]
                    y_train = y_train.astype(float)
                    # est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0,
                    #                                 loss='ls').fit(X_train, y_train)
                    n_neighbors = 15
                    weights = 'distance'
                    reg = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
                    pred = reg.fit(X_train, y_train).predict(X_test)
                    ret.append(pred)
            else:
                X_train = X_train_df.iloc[:, 6:12]
                X_train = X_train.astype(float)
                X_test = X_test_df.iloc[:, 6:12]
                X_test = X_test.astype(float)
                for i in range(6, 12):
                    y_train = y_train_df.iloc[:, i]
                    y_train = y_train.astype(float)
                    # est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0,
                    #                                 loss='ls').fit(X_train, y_train)
                    n_neighbors = 15
                    weights = 'distance'
                    reg = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
                    pred = reg.fit(X_train, y_train).predict(X_test)
                    ret.append(pred)
        predict = pd.DataFrame(np.array(ret).T, columns=['travel_time' + str(index) for index in range(1, 13)])
        predict['intersection_id'] = intersection_id
        predict['tollgate_id'] = tollgate_id
        predict['date'] = X_test_df['date']
        predict = to_submit_format_df(predict)
        predict[['tollgate_id']] = predict[['tollgate_id']].astype(int)
        predict[['avg_travel_time']] = predict[['avg_travel_time']].astype(float)
        predict_df = predict_df.append(predict, ignore_index=True)

    print task1_eva_metrics(predict_df.copy(), pd.read_csv('10.11-10.17-8:00-10:00.csv', header=0))


def predict():
    predict_df = pd.DataFrame(columns=['intersection_id', 'tollgate_id', 'time_window', 'avg_travel_time'])
    roads = [['A', 2], ['A', 3], ['B', 1], ['B', 3], ['C', 1], ['C', 3]]
    test_df = read_testing_data(testing_file)
    for trajectory in roads:
        intersection_id = trajectory[0]
        tollgate_id = trajectory[1]
        train_df = read_training_data(training_file)
        train_df = train_df[(train_df.intersection_id == intersection_id) & (train_df.tollgate_id == tollgate_id)]
        X_train_df = feature_extra(train_df)
        y_train_df = label_extra(train_df)

        X_test_df = feature_extra(test_df, start_date='2016-10-10', end_date='2016-10-18')
        X_train_df, y_train_df = data_process(X_train_df, y_train_df)
        ret = []
        for flag in [True, False]:
            if flag:
                X_train = X_train_df.iloc[:, 0:6]
                X_train = X_train.astype(float)
                X_test = X_test_df.iloc[:, 0:6]
                X_test = X_test.astype(float)
                for i in range(6):
                    y_train = y_train_df.iloc[:, i]
                    y_train = y_train.astype(float)
                    # est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0,
                    #                                 loss='ls').fit(X_train, y_train)
                    n_neighbors = 15
                    weights = 'distance'
                    reg = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
                    pred = reg.fit(X_train, y_train).predict(X_test)
                    ret.append(pred)
            else:
                X_train = X_train_df.iloc[:, 6:12]
                X_train = X_train.astype(float)
                X_test = X_test_df.iloc[:, 6:12]
                X_test = X_test.astype(float)
                for i in range(6, 12):
                    y_train = y_train_df.iloc[:, i]
                    y_train = y_train.astype(float)
                    # est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0,
                    #                                 loss='ls').fit(X_train, y_train)
                    n_neighbors = 15
                    weights = 'distance'
                    reg = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
                    pred = reg.fit(X_train, y_train).predict(X_test)
                    ret.append(pred)
        predict = pd.DataFrame(np.array(ret).T, columns=['travel_time' + str(index) for index in range(1, 13)])
        predict['intersection_id'] = intersection_id
        predict['tollgate_id'] = tollgate_id
        predict['date'] = X_test_df['date']
        predict = to_submit_format_df(predict)
        predict[['tollgate_id']] = predict[['tollgate_id']].astype(int)
        predict[['avg_travel_time']] = predict[['avg_travel_time']].astype(float)
        predict_df = predict_df.append(predict, ignore_index=True)
