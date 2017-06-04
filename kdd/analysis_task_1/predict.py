# -*- coding: utf-8 -*-

"""
对10.25-10.31的测试数据进行预测
"""

import datetime
import pandas as pd
import numpy as np
import re
from sklearn import neighbors

training_file1 = "/home/sunwill/PycharmProjects/kdd/datasets/dataSets/training/trajectories(table 5)_training.csv"
training_file2 = "/home/sunwill/PycharmProjects/kdd/datasets/dataSet_phase2/dataSet_phase2/trajectories(table_5)_training2.csv"
testing_file = "/home/sunwill/PycharmProjects/kdd/datasets/dataSet_phase2/dataSet_phase2/trajectories(table 5)_test2.csv"


def read_training_data(filename1, filename2):
    training_data1 = pd.read_csv(filename1, header=0)
    training_data2 = pd.read_csv(filename2, header=0)
    training_data = training_data1.append(training_data2, ignore_index=True)
    return training_data


def read_testing_data(filename):
    return pd.read_csv(filename, header=0)


def drop_spec_date(data, model='model2', start_date='2016-07-18', end_date='2016-10-24'):
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
        return data
        # date_window = [[start_date, '2016-10-01'], ['2016-10-08', end_date]]
    else:
        date_window = [['2016-10-18 00:00:00', '2016-10-24 23:59:59']]
    data_df = pd.DataFrame(columns=data.columns)
    for item in date_window:
        _df = data[(data.starting_time >= item[0]) & (data.starting_time <= item[1])]
        data_df = data_df.append(_df, ignore_index=True)
    return data

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
    data.to_csv(filename, index=False)


def get_avg_time(flag1=True, flag2=True):
    """
    :param model:
    :param flag1:
    :param flag2:
    :return:
    """
    data = pd.read_csv('/home/sunwill/PycharmProjects/kdd/datasets/dataSet_phase2/dataSet_phase2/trajectories(table_5)_training2.csv', header=0)
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


def feature_extract(train_df, flag=True, start_date='2016-07-18', end_date='2016-10-24'):
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
    # data_df = data_df.fillna(data_df.mean())
    return data_df


def label_extract(train_df, flag=True, start_date='2016-07-18', end_date='2016-10-24'):
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
    # data_df = data_df.fillna(data_df.mean())
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


def predict():
    predict_df = pd.DataFrame(columns=['intersection_id', 'tollgate_id', 'time_window', 'avg_travel_time'])
    # roads = [['A', 2], ['A', 3], ['B', 3, True], ['C', 3]]
    # for trajectory in roads:
    #     if len(trajectory) == 3:
    #         sec = [trajectory[-1]]
    #     else:
    #         sec = [True, False]
    #     for flag in sec:  # 分上班和下班，上班为True,下班为False
    #         train_df = read_training_data(training_file1, training_file2)
    #         test_df = read_testing_data(testing_file)
    #         train_df = drop_spec_date(train_df)
    #         train_df = train_df[(train_df.intersection_id == trajectory[0]) & (train_df.tollgate_id == trajectory[1])]
    #         test_df = test_df[(test_df.intersection_id == trajectory[0]) & (test_df.tollgate_id == trajectory[1])]
    #
    #         X_train_df = feature_extract(train_df, flag=flag, start_date='2016-07-18', end_date='2016-10-24')
    #         y_train_df = label_extract(train_df, flag=flag, start_date='2016-07-18', end_date='2016-10-24')
    #         X_train_df, y_train_df = data_process(X_train_df, y_train_df)
    #         # print X_train_df
    #         # print y_train_df
    #         X_train = X_train_df.iloc[:, -6:]
    #         y_train = y_train_df.iloc[:, -6:]
    #
    #         X_test_df = feature_extract(test_df, flag=flag, start_date='2016-10-24', end_date='2016-10-31')
    #         X_test = X_test_df.iloc[:, -6:]
    #
    #         n_neighbors = 15
    #         weights = 'distance'
    #         reg = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    #         reg.fit(X_train, y_train)
    #         pred = reg.predict(X_test)
    #
    #         pred_df = pd.DataFrame(pred, columns=X_test_df.columns[3:])
    #         _intersection_arr = X_test_df['intersection_id']
    #         _tollgate_id_arr = X_test_df['tollgate_id']
    #         _date_arr = X_test_df['date']
    #         pred_df.insert(0, 'date', np.array(_date_arr))
    #         pred_df.insert(0, 'tollgate_id', np.array(_tollgate_id_arr))
    #         pred_df.insert(0, 'intersection_id', np.array(_intersection_arr))
    #
    #         pred_df = to_submit_format_df(pred_df, flag=flag)
    #         predict_df = predict_df.append(pred_df, ignore_index=True)

    date_array = ['2016-10-25', '2016-10-26', '2016-10-27', '2016-10-28', '2016-10-29', '2016-10-30', '2016-10-31']
    # roads = [['B', 1], ['B', 3, False], ['C', 1]]
    roads = [['A', 2], ['A', 3], ['B', 1], ['B', 3], ['C', 1], ['C', 3]]
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
            traj = get_avg_time(flag1=flag, flag2=False)
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

    write_to_file(predict_df, '/home/sunwill/PycharmProjects/kdd/result_6.1.csv')
