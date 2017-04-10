# -*- coding: utf-8 -*-

"""
task1_model_2:回归模型
训练数据：2016-07-19至2016-10-17的三个月车辆轨迹数据以及2016-09-17至2016-10-17一个月的交通流量数据
构建特征工程
输入空间来自三个月前两个小时6:00:00-8:00:00和15:00:00-17:00:00，输出空间为对应的后两小时8:00:00-10:00：00和17:00:00-19:00:00
在测试数据中按相同方式构建特征工程
测试数据（预测数据）：2016-10-18至2016-10-24一个星期的8:00:00-10:00：00和17:00:00-19:00:00
"""
import os
import datetime
from os import path
import pandas as pd
import numpy as np
import re
import xgboost as xgb
from sklearn import neighbors
from kdd.metrics import task1_eva_metrics
from sklearn import linear_model
from sklearn.multioutput import MultiOutputRegressor

PROJECT_PATH = os.path.dirname(os.path.abspath(__name__))
dir_path = path.join(PROJECT_PATH, "datasets/dataSets/")
train_data_path = path.join(dir_path, "training/")
test_data_path = path.join(dir_path, "testing_phase1/")


def read_training_data(filename):
    filepath = train_data_path + filename
    return pd.read_csv(filepath, header=0)


def read_testing_data(filename):
    filepath = test_data_path + filename
    return pd.read_csv(filepath, header=0)


def drop_spec_date(data, model='model2', start_date='2016-07-18', end_date='2016-10-18'):
    """
    日期预处理，剔除十一等特殊日期
    2016.7.19-2016.9.29和2016.10.8-2016.10.17
    :param model:
    :param data:
    :param start_date: 输入数据需要提前一天，且是string类型
    :param end_date: 输出数据为当天日期，string类型
    :return:
    """
    if model == 'model2':
        date_window = [[start_date, '2016-10-01'], ['2016-10-08', end_date]]
    else:
        date_window = [[start_date, '2016-09-15 00:00:00']]
    data_df = pd.DataFrame(columns=data.columns)
    for item in date_window:
        _df = data[(data.starting_time >= item[0]) & (data.starting_time <= item[1])]
        data_df = data_df.append(_df, ignore_index=True)
    return data_df


def to_submit_format_df(df, flag=True):
    """
    :param flag:
    :param df:
    :param flag1: 是否上午，上午为true,下午为False
    :return:
    """
    if flag:
        time_inteval = [
            ['08:00:00', '08:20:00'], ['08:20:00', '08:40:00'], ['08:40:00', '09:00:00'],
            ['09:00:00', '09:20:00'], ['09:20:00', '09:40:00'], ['09:40:00', '10:00:00'],
        ]
    else:
        time_inteval = [
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


def write_to_file(data, filename):
    """
    将结果写入文件
    :param filename:
    :param data:
    :return:
    """
    data.to_csv(PROJECT_PATH + "/" + filename, index=False)


def get_avg_time(model="model2", flag1=True, flag2=True):
    """
    :param model:
    :param flag1:
    :param flag2:
    :return:
    """
    data = read_training_data('trajectories(table 5)_training.csv')
    data = drop_spec_date(data, model)
    if flag1:
        if flag2:
            time_inteval = [
                ['06:00:00', '06:20:00'], ['06:20:00', '06:40:00'], ['06:40:00', '07:00:00'],
                ['07:00:00', '07:20:00'], ['07:20:00', '07:40:00'], ['07:40:00', '08:00:00']]
        else:
            time_inteval = [
                ['08:00:00', '08:20:00'], ['08:20:00', '08:40:00'], ['08:40:00', '09:00:00'],
                ['09:00:00', '09:20:00'], ['09:20:00', '09:40:00'], ['09:40:00', '10:00:00']]

    else:
        if flag2:
            time_inteval = [
                ['15:00:00', '15:20:00'], ['15:20:00', '15:40:00'], ['15:40:00', '16:00:00'],
                ['16:00:00', '16:20:00'], ['16:20:00', '16:40:00'], ['16:40:00', '17:00:00']]
        else:
            time_inteval = [
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
                for window in range(1, 7):
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
        for time_window in range(1, 7):
            travel_time = 0
            for link_id in trajectory:
                travel_time += link_time[str(link_id)]['travel_time' + str(time_window)]
            traj[key]['travel_time' + str(time_window)] = travel_time
    return traj


def set_missing_data(df, flag1=True, flag2=True):
    """
    用trajectory的平均时间代替缺失值
    :param df:
    :param flag1: 上午为True,下午为False
    :param flag2: 前两小时为True,后两小时为False
    :return:
    """
    traj = get_avg_time(flag1, flag2)
    col_names = ['travel_time' + str(x) for x in range(1, 7)]
    result = pd.DataFrame(columns=df.columns)
    for index, row in df.iterrows():
        inte_id = row['intersection_id']
        to_id = row['tollgate_id']
        for col in col_names:
            if row[col] != row[col]:  # 为缺失值
                row[col] = traj[str(inte_id) + '->' + str(int(to_id))][col]
        result = result.append(pd.DataFrame(np.array([row]), columns=result.columns), ignore_index=True)
    return result


def feature_extract(train_df, flag=True, start_date='2016-07-18', end_date='2016-10-10'):
    """
    提取特征
    :param train_df:
    :param flag:
    :param start_date:
    :param end_date:
    :return:
    """
    if flag:
        time_window = [['06:00:00', '06:20:00'], ['06:20:00', '06:40:00'], ['06:40:00', '07:00:00'],
                       ['07:00:00', '07:20:00'], ['07:20:00', '07:40:00'], ['07:40:00', '08:00:00']]
    else:
        time_window = [['15:00:00', '15:20:00'], ['15:20:00', '15:40:00'], ['15:40:00', '16:00:00'],
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

    data_df = set_missing_data(data_df, flag1=flag, flag2=True)

    return data_df


def label_extract(train_df, flag=True, start_date='2016-07-18', end_date='2016-10-10'):
    """
    提取标签值
    :param train_df:
    :param flag:
    :param start_date:
    :param end_date:
    :return:
    """
    if flag:
        time_window = [['08:00:00', '08:20:00'], ['08:20:00', '08:40:00'], ['08:40:00', '09:00:00'],
                       ['09:00:00', '09:20:00'], ['09:20:00', '09:40:00'], ['09:40:00', '10:00:00']]
    else:
        time_window = [['17:00:00', '17:20:00'], ['17:20:00', '17:40:00'], ['17:40:00', '18:00:00'],
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
    data_df = set_missing_data(data_df, flag1=flag, flag2=False)
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


def local_test():
    """
    本地测试,调参
    :return:
    A->2:
            0.207286293321
        0.161001285742
    A->3
            0.267706847717
        0.132676985484
    B->1
        0.156456409916
            0.232996593942
    B->3
            0.196764506936
            0.339868798445
    C->1
        0.133456726619
        0.183167886662
    C->3
            0.221149507591
            0.208706056411
    模型融合之后,0.15218938889686356
    """
    predict_df = pd.DataFrame(columns=['intersection_id', 'tollgate_id', 'time_window', 'avg_travel_time'])
    actual_df = pd.DataFrame(columns=['intersection_id', 'tollgate_id', 'time_window', 'avg_travel_time'])
    roads = [['A', 2, False], ['A', 3, False], ['B', 1, True], ['C', 1]]
    training_file = 'trajectories(table 5)_training.csv'
    for trajectory in roads:
        if len(trajectory) == 3:
            sec = [trajectory[-1]]
        else:
            sec = [True, False]
        for flag in sec:  # 分上班和下班，上班为True,下班为False
            train_df = read_training_data(training_file)
            train_df = drop_spec_date(train_df)
            train_df = train_df[(train_df.intersection_id == trajectory[0]) & (train_df.tollgate_id == trajectory[1])]

            X_train_df = feature_extract(train_df, flag=flag, start_date='2016-07-18', end_date='2016-10-10')
            y_train_df = label_extract(train_df, flag=flag, start_date='2016-07-18', end_date='2016-10-10')

            X_train_df, y_train_df = data_process(X_train_df, y_train_df)

            X_train = X_train_df.iloc[:, -6:]
            y_train = y_train_df.iloc[:, -6:]

            X_test_df = feature_extract(train_df, flag=flag, start_date='2016-10-10', end_date='2016-10-17')
            y_test_df = label_extract(train_df, flag=flag, start_date='2016-10-10', end_date='2016-10-17')
            X_test_df, y_test_df = data_process(X_test_df, y_test_df)

            X_test = X_test_df.iloc[:, -6:]

            n_neighbors = 10
            weights = 'distance'
            reg = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
            pred = reg.fit(X_train, y_train).predict(X_test)

            pred_df = pd.DataFrame(pred, columns=X_test_df.columns[3:])
            _intersection_arr = X_test_df['intersection_id']
            _tollgate_id_arr = X_test_df['tollgate_id']
            _date_arr = X_test_df['date']
            pred_df.insert(0, 'date', np.array(_date_arr))
            pred_df.insert(0, 'tollgate_id', np.array(_tollgate_id_arr))
            pred_df.insert(0, 'intersection_id', np.array(_intersection_arr))

            pred_df = to_submit_format_df(pred_df, flag=flag)
            act_df = to_submit_format_df(y_test_df, flag=flag)
            predict_df = predict_df.append(pred_df, ignore_index=True)
            actual_df = actual_df.append(act_df, ignore_index=True)
    print task1_eva_metrics(predict_df.copy(), actual_df.copy())

    date_array = ['2016-10-11', '2016-10-12', '2016-10-13', '2016-10-14', '2016-10-15', '2016-10-16', '2016-10-17']
    roads = [['A', 2, True], ['A', 3, True], ['B', 1, False], ['B', 3], ['C', 3]]
    # roads = [['A', 2], ['A', 3], ['B', 1], ['B', 3], ['C', 1], ['C', 3]]

    # go_to_work = [['06:00:00', '06:20:00'], ['06:20:00', '06:40:00'], ['06:40:00', '07:00:00'], ['07:00:00', '07:20:00'],
    # ['07:20:00', '07:40:00'], ['07:40:00', '08:00:00']]
    # leave_for_work = [['15:00:00', '15:20:00'], ['15:20:00', '15:40:00'], ['15:40:00', '16:00:00'], ['16:00:00', '16:20:00'],
    # ['16:20:00', '16:40:00'], ['16:40:00', '17:00:00']]
    go_to_work = [['08:00:00', '08:20:00'], ['08:20:00', '08:40:00'], ['08:40:00', '09:00:00'],
                  ['09:00:00', '09:20:00'], ['09:20:00', '09:40:00'], ['09:40:00', '10:00:00']]
    leave_for_work = [['17:00:00', '17:20:00'], ['17:20:00', '17:40:00'], ['17:40:00', '18:00:00'],
                      ['18:00:00', '18:20:00'], ['18:20:00', '18:40:00'], ['18:40:00', '19:00:00']]
    for idx, road in enumerate(roads):
        if len(road) == 3:
            sec = [road[-1]]
        else:
            sec = [True, False]
        for flag in sec:
            traj = get_avg_time(model='model4', flag1=flag, flag2=False)
            if flag:
                time_window = go_to_work
            else:
                time_window = leave_for_work
            for day in date_array:
                for win_idx in range(1, 7):
                    row = [roads[idx][0], roads[idx][1],
                           '[' + day + ' ' + time_window[win_idx - 1][0] + ',' + day + ' ' +
                           time_window[win_idx - 1][1] + ')']
                    travel_time = traj[str(road[0]) + '->' + str(int(road[1]))]['travel_time' + str(win_idx)]
                    row.append(travel_time)
                    predict_df = predict_df.append(pd.DataFrame(np.array([row]), columns=predict_df.columns),
                                                   ignore_index=True)

    predict_df[['tollgate_id']] = predict_df[['tollgate_id']].astype(int)
    predict_df[['avg_travel_time']] = predict_df[['avg_travel_time']].astype(float)

    print task1_eva_metrics(predict_df.copy(), pd.read_csv('10.11-10.17-8:00-10:00.csv', header=0))


def predict():
    predict_df = pd.DataFrame(columns=['intersection_id', 'tollgate_id', 'time_window', 'avg_travel_time'])
    roads = [['A', 2, False], ['A', 3, False], ['B', 1, True], ['C', 1]]
    training_file = 'trajectories(table 5)_training.csv'
    testing_file = 'trajectories(table 5)_test1.csv'
    for trajectory in roads:
        if len(trajectory) == 3:
            sec = [trajectory[-1]]
        else:
            sec = [True, False]
        for flag in sec:  # 分上班和下班，上班为True,下班为False
            train_df = read_training_data(training_file)
            test_df = read_testing_data(testing_file)
            train_df = drop_spec_date(train_df)
            train_df = train_df[(train_df.intersection_id == trajectory[0]) & (train_df.tollgate_id == trajectory[1])]
            test_df = test_df[(test_df.intersection_id == trajectory[0]) & (test_df.tollgate_id == trajectory[1])]

            X_train_df = feature_extract(train_df, flag=flag, start_date='2016-07-18', end_date='2016-10-17')
            y_train_df = label_extract(train_df, flag=flag, start_date='2016-07-18', end_date='2016-10-17')

            X_train_df, y_train_df = data_process(X_train_df, y_train_df)

            X_train = X_train_df.iloc[:, -6:]
            y_train = y_train_df.iloc[:, -6:]

            X_test_df = feature_extract(test_df, flag=flag, start_date='2016-10-17', end_date='2016-10-24')

            X_test = X_test_df.iloc[:, -6:]

            n_neighbors = 10
            weights = 'distance'
            reg = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
            pred = reg.fit(X_train, y_train).predict(X_test)

            pred_df = pd.DataFrame(pred, columns=X_test_df.columns[3:])
            _intersection_arr = X_test_df['intersection_id']
            _tollgate_id_arr = X_test_df['tollgate_id']
            _date_arr = X_test_df['date']
            pred_df.insert(0, 'date', np.array(_date_arr))
            pred_df.insert(0, 'tollgate_id', np.array(_tollgate_id_arr))
            pred_df.insert(0, 'intersection_id', np.array(_intersection_arr))

            pred_df = to_submit_format_df(pred_df, flag=flag)
            predict_df = predict_df.append(pred_df, ignore_index=True)

    date_array = ['2016-10-18', '2016-10-19', '2016-10-20', '2016-10-21', '2016-10-22', '2016-10-23', '2016-10-24']
    roads = [['A', 2, True], ['A', 3, True], ['B', 1, False], ['B', 3], ['C', 3]]
    go_to_work = [['08:00:00', '08:20:00'], ['08:20:00', '08:40:00'], ['08:40:00', '09:00:00'],
                  ['09:00:00', '09:20:00'], ['09:20:00', '09:40:00'], ['09:40:00', '10:00:00']]
    leave_for_work = [['17:00:00', '17:20:00'], ['17:20:00', '17:40:00'], ['17:40:00', '18:00:00'],
                      ['18:00:00', '18:20:00'], ['18:20:00', '18:40:00'], ['18:40:00', '19:00:00']]
    for idx, road in enumerate(roads):
        if len(road) == 3:
            sec = [road[-1]]
        else:
            sec = [True, False]
        for flag in sec:
            traj = get_avg_time(model='model4', flag1=flag, flag2=False)
            if flag:
                time_window = go_to_work
            else:
                time_window = leave_for_work
            for day in date_array:
                for win_idx in range(1, 7):
                    row = [roads[idx][0], roads[idx][1],
                           '[' + day + ' ' + time_window[win_idx - 1][0] + ',' + day + ' ' +
                           time_window[win_idx - 1][1] + ')']
                    travel_time = traj[str(road[0]) + '->' + str(int(road[1]))]['travel_time' + str(win_idx)]
                    row.append(travel_time)
                    predict_df = predict_df.append(pd.DataFrame(np.array([row]), columns=predict_df.columns),
                                                   ignore_index=True)

    predict_df[['tollgate_id']] = predict_df[['tollgate_id']].astype(int)
    predict_df[['avg_travel_time']] = predict_df[['avg_travel_time']].astype(float)

    write_to_file(predict_df, 'task_1_Model_2_combine_model_4_submit.csv')
