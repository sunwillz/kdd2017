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
    filepath = train_data_path+filename
    return pd.read_csv(filepath, header=0)

def read_testing_data(filename):
    filepath = test_data_path+filename
    return pd.read_csv(filepath, header=0)

def drop_spec_date(data, start_date = '2016-07-18', end_date = '2016-10-18'):
    """
    日期预处理，剔除十一等特殊日期
    2016.7.19-2016.9.29和2016.10.8-2016.10.17
    :param data:
    :param start_date: 输入数据需要提前一天，且是string类型
    :param end_date: 输出数据为当天日期，string类型
    :return:
    """
    date_window = [[start_date, '2016-09-30'], ['2016-10-08', end_date]]
    data_df = pd.DataFrame(columns=data.columns)
    for item in date_window:
        _df = data[(data.starting_time >= item[0]) & (data.starting_time <= item[1])]
        data_df = data_df.append(_df, ignore_index=True)
    return data_df

def set_missing_data(df):
    """
    处理缺失值，用均值代替
    :param df:
    :return:
    """
    return df.fillna(df.mean())


def form_training_data(data, start_date='2016-07-18', end_date='2016-10-10'):
    """
    格式化训练数据
    :param data:
    :param start_date: 起始日期，提前一天
    :param end_date: 终止日期，当天
    :return:
    """
    time_inteval_known = [
        ['06:00:00', '06:20:00'], ['06:20:00', '06df:40:00'], ['06:40:00', '07:00:00'], ['07:00:00','07:20:00'], ['07:20:00', '07:40:00'], ['07:40:00','08:00:00'],
        ['15:00:00', '15:20:00'], ['15:20:00', '15:40:00'], ['15:40:00', '16:00:00'], ['16:00:00', '16:20:00'], ['16:20:00', '16:40:00'], ['16:40:00', '17:00:00']

    ]
    time_inteval_pred = [
        ['08:00:00', '08:20:00'], ['08:20:00', '08:40:00'], ['08:40:00', '09:00:00'], ['09:00:00', '09:20:00'], ['09:20:00', '09:40:00'], ['09:40:00', '10:00:00'],
        ['17:00:00', '17:20:00'], ['17:20:00', '17:40:00'], ['17:40:00', '18:00:00'], ['18:00:00', '18:20:00'], ['18:20:00', '18:40:00'], ['18:40:00', '19:00:00']

    ]
    # start_date='2016-07-18'
    # end_date='2016-10-10'
    date1 = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    date2 = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    i = datetime.timedelta(days=1)
    columns_names = ["intersection_id", "tollgate_id", "date", "travel_time"]
    columns_names1 = ['intersection_id', 'tollgate_id', 'date']
    for inde in range(12):
       columns_names1.append('travel_time'+str(inde+1))
    temp_data_df_known = pd.DataFrame(columns=columns_names)
    temp_data_df_pred = pd.DataFrame(columns=columns_names)

    while i <= (date2-date1):
        cur_date = (date1+i).strftime('%Y-%m-%d')
        for index, inteval in enumerate(time_inteval_known):
            df = data[(data.starting_time >= cur_date + " "+inteval[0]) & (data.starting_time <= cur_date + " "+inteval[1])]
            df = df.groupby(['intersection_id', 'tollgate_id']).mean().reset_index()
            df['time_window'] = "[" + cur_date + " " + inteval[0] + "," + cur_date + " " + inteval[1] + ")"
            df['date'] = cur_date
            df['travel_time'+str(index+1)] = df.travel_time
            new_data_df = df[['intersection_id', 'tollgate_id', 'date', 'travel_time', 'travel_time'+str(index+1)]]
            temp_data_df_known = temp_data_df_known.append(new_data_df, ignore_index=True)
        for index, inteval in enumerate(time_inteval_pred):
            df = data[(data.starting_time >= cur_date + " " + inteval[0]) & (data.starting_time <= cur_date + " " + inteval[1])]
            df = df.groupby(['intersection_id', 'tollgate_id']).mean().reset_index()
            df['time_window'] = "["+cur_date+" " + inteval[0] + "," + cur_date + " " + inteval[1] + ")"
            df['date'] = cur_date
            df['travel_time' + str(index + 1)] = df.travel_time
            new_data_df = df[['intersection_id', 'tollgate_id', 'date', 'travel_time', 'travel_time'+str(index+1)]]
            temp_data_df_pred = temp_data_df_pred.append(new_data_df, ignore_index=True)
        i += datetime.timedelta(days=1)
    temp_data_df_known = temp_data_df_known.groupby(['intersection_id', 'tollgate_id', 'date']).sum().reset_index()
    temp_data_df_pred = temp_data_df_pred.groupby(['intersection_id', 'tollgate_id', 'date']).sum().reset_index()
    # print temp_data_df_known #此处注意出现很多空值，需要谨慎处理，否则噪声带来的误差会很大
    del temp_data_df_known['travel_time']
    del temp_data_df_pred['travel_time']
    temp_data_df_known = set_missing_data(temp_data_df_known)
    temp_data_df_pred = set_missing_data(temp_data_df_pred)
    #重排序列
    for idx in range(12, 0, -1):
        col = temp_data_df_known.pop('travel_time'+str(idx))
        temp_data_df_known.insert(0, 'travel_time'+str(idx), col)
        col = temp_data_df_pred.pop('travel_time' + str(idx))
        temp_data_df_pred.insert(0, 'travel_time' + str(idx), col)
    return temp_data_df_known, temp_data_df_pred

def form_testing_data(data, start_date, end_date):
    """
    格式化测试数据
    :param data:
    :param start_date: 起始日期，提前一天,起始时间是'2016-10-18',则要输入'2016-10-17'
    :param end_date: 终止日期，当天，如'2016-10-24'
    :return:
    """
    time_window = [
        ['06:00:00', '06:20:00'], ['06:20:00', '06:40:00'], ['06:40:00', '07:00:00'], ['07:00:00', '07:20:00'],
        ['07:20:00', '07:40:00'], ['07:40:00', '08:00:00'],
        ['15:00:00', '15:20:00'], ['15:20:00', '15:40:00'], ['15:40:00', '16:00:00'], ['16:00:00', '16:20:00'],
        ['16:20:00', '16:40:00'], ['16:40:00', '17:00:00']

    ]

    # start_date='2016-07-18'
    # end_date='2016-10-10'
    date1 = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    date2 = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    i = datetime.timedelta(days=1)
    columns_names = ["intersection_id", "tollgate_id", "date", "travel_time"]
    columns_names1 = ['intersection_id', 'tollgate_id', 'date']
    for inde in range(12):
        columns_names1.append('travel_time' + str(inde + 1))
    temp_data_df_known = pd.DataFrame(columns=columns_names)

    while i <= (date2 - date1):
        cur_date = (date1 + i).strftime('%Y-%m-%d')
        for index, inteval in enumerate(time_window):
            df = data[(data.starting_time >= cur_date + " " + inteval[0]) & (
            data.starting_time <= cur_date + " " + inteval[1])]
            df = df.groupby(['intersection_id', 'tollgate_id']).mean().reset_index()
            df['time_window'] = "[" + cur_date + " " + inteval[0] + "," + cur_date + " " + inteval[1] + ")"
            df['date'] = cur_date
            df['travel_time' + str(index + 1)] = df.travel_time
            new_data_df = df[['intersection_id', 'tollgate_id', 'date', 'travel_time', 'travel_time' + str(index + 1)]]
            temp_data_df_known = temp_data_df_known.append(new_data_df, ignore_index=True)
        i += datetime.timedelta(days=1)
    temp_data_df_known = temp_data_df_known.groupby(['intersection_id', 'tollgate_id', 'date']).sum().reset_index()
    del temp_data_df_known['travel_time']
    temp_data_df_known = set_missing_data(temp_data_df_known)
    for idx in range(12, 0, -1):
        col = temp_data_df_known.pop('travel_time'+str(idx))
        temp_data_df_known.insert(0, 'travel_time'+str(idx), col)

    return temp_data_df_known

def to_submit_format_df(df,flag = False):
    """
    转成提交文件的格式
    :param actual_df:
    :param pred_df:
    :return:
    """
    if flag:
        time_inteval = [
            ['08:00:00', '08:20:00'], ['08:20:00', '08:40:00'], ['08:40:00', '09:00:00'], ['09:00:00', '09:20:00'], ['09:20:00', '09:40:00'], ['09:40:00', '10:00:00'],
            ['17:00:00', '17:20:00'], ['17:20:00', '17:40:00'], ['17:40:00', '18:00:00'], ['18:00:00', '18:20:00'], ['18:20:00', '18:40:00'], ['18:40:00', '19:00:00']

        ]
    else:
        time_inteval = [
            ['06:00:00', '06:20:00'], ['06:20:00', '06:40:00'], ['06:40:00', '07:00:00'], ['07:00:00', '07:20:00'], ['07:20:00', '07:40:00'], ['07:40:00', '08:00:00'],
            ['15:00:00', '15:20:00'], ['15:20:00', '15:40:00'], ['15:40:00', '16:00:00'], ['16:00:00', '16:20:00'], ['16:20:00', '16:40:00'], ['16:40:00', '17:00:00']
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
                time_window = "["+_date+" "+time_inteval[idx-1][0]+","+_date+" "+time_inteval[idx-1][1]+")"
                tp_list.append(time_window)
                tp_list.append(row[col_name])
                result_df.append(tp_list)

    return pd.DataFrame(result_df, columns=['intersection_id', 'tollgate_id', 'time_window', 'avg_travel_time'])


def write_to_file(data,filename):
    """
    将结果写入文件
    :param data:
    :return:
    """
    data.to_csv(PROJECT_PATH+"/"+filename, index=False)

def knn_reg():

    training_file = 'trajectories(table 5)_training.csv'

    #共有6条路线，A->2,A->3,B->1,B->3,C->1,C->3,对每条路线建立回归模型
    trajectories = [['A', 2], ['A', 3], ['B', 1], ['B', 3], ['C', 1], ['C', 3]]
    actual_df = pd.DataFrame(columns=['intersection_id', 'tollgate_id', 'time_window', 'avg_travel_time'])
    pred_df = pd.DataFrame(columns=['intersection_id', 'tollgate_id', 'time_window', 'avg_travel_time'])
    for trajectory in trajectories:
        train_df = read_training_data(training_file)
        train_df = drop_spec_date(train_df)
        train_df = train_df[(train_df.intersection_id == trajectory[0]) & (train_df.tollgate_id == trajectory[1])]
        train_df_known, train_df_pred = form_training_data(train_df, start_date='2016-07-18', end_date='2016-10-10')
        test_df_known, test_df_pred = form_training_data(train_df, start_date='2016-10-10', end_date='2016-10-17')

        #由于训练集中缺少2016.08.26 B->1 8:00-10:00和17:00-19:00的数据，因此在train_df_knownz中需要去除当天B->1的数据，以免噪声干扰
        if trajectory[0] == 'B' and trajectory[1] == 1:
            _idx = train_df_known.index[(train_df_known.intersection_id == 'B') & (train_df_known.tollgate_id == 1) & (train_df_known.date == '2016-08-26')]
            train_df_known_above = train_df_known.loc[:int(_idx.values)-1]
            train_df_known_below = train_df_known.loc[int(_idx.values)+1:]
            train_df_known = train_df_known_above.append(train_df_known_below, ignore_index=True)

        X = train_df_known.drop(["intersection_id", "tollgate_id", "date"], axis=1)
        y = train_df_pred.drop(["intersection_id", "tollgate_id", "date"], axis=1)
        T = test_df_known.drop(["intersection_id", "tollgate_id", "date"], axis=1)
        T['travel_time1'] = T.mean(axis=1)
        n_neighbors = 10
        weights = 'uniform'
        reg = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
        # reg = MultiOutputRegressor(linear_model.LassoLars(alpha=.1))
        pred_ret = reg.fit(X, y).predict(T)

        ret_list = pd.DataFrame(pred_ret, columns=train_df_known.columns[:12])
        _intersection_arr = test_df_known['intersection_id']
        _tollgate_id_arr = test_df_known['tollgate_id']
        _date_arr = test_df_known['date']
        ret_list.insert(0, 'date', np.array(_date_arr))
        ret_list.insert(0, 'tollgate_id', np.array(_tollgate_id_arr))
        ret_list.insert(0, 'intersection_id', np.array(_intersection_arr))

        actual_df = actual_df.append(to_submit_format_df(test_df_pred), ignore_index=True)
        actual_df.to_csv('validation.txt', index=False)
        pred_df = pred_df.append(to_submit_format_df(ret_list), ignore_index=True)
    score = task1_eva_metrics(pred_df, actual_df)
    return score

def predict_and_write():

    training_file = 'trajectories(table 5)_training.csv'
    testing_file = 'trajectories(table 5)_test1.csv'

    # 共有6条路线，A->2,A->3,B->1,B->3,C->1,C->3,对每条路线建立回归模型
    trajectories = [['A', 2], ['A', 3], ['B', 1], ['B', 3], ['C', 1], ['C', 3]]
    actual_df = pd.DataFrame(columns=['intersection_id', 'tollgate_id', 'time_window', 'avg_travel_time'])
    pred_df = pd.DataFrame(columns=['intersection_id', 'tollgate_id', 'time_window', 'avg_travel_time'])
    for trajectory in trajectories:
        train_df = read_training_data(training_file)
        test_df = read_testing_data(testing_file)
        train_df = drop_spec_date(train_df)
        train_df = train_df[(train_df.intersection_id == trajectory[0]) & (train_df.tollgate_id == trajectory[1])]
        test_df = test_df[(test_df.intersection_id == trajectory[0]) & (test_df.tollgate_id == trajectory[1])]
        train_df_known, train_df_pred = form_training_data(train_df, start_date='2016-07-18', end_date='2016-10-17')
        test_df_known = form_testing_data(test_df, start_date='2016-10-17', end_date='2016-10-24')

        # 由于训练集中缺少2016.08.26 B->1 8:00-10:00和17:00-19:00的数据，因此在train_df_knownz中需要去除当天B->1的数据，以免噪声干扰
        if trajectory[0] == 'B' and trajectory[1] == 1:
            _idx = train_df_known.index[(train_df_known.intersection_id == 'B') & (train_df_known.tollgate_id == 1) & (
            train_df_known.date == '2016-08-26')]
            train_df_known_above = train_df_known.loc[:int(_idx.values) - 1]
            train_df_known_below = train_df_known.loc[int(_idx.values) + 1:]
            train_df_known = train_df_known_above.append(train_df_known_below, ignore_index=True)

        X = train_df_known.drop(["intersection_id", "tollgate_id", "date"], axis=1)
        y = train_df_pred.drop(["intersection_id", "tollgate_id", "date"], axis=1)
        T = test_df_known.drop(["intersection_id", "tollgate_id", "date"], axis=1)
        T['travel_time1'] = T.mean(axis=1)
        n_neighbors = 10
        weights = 'distance'
        reg = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
        pred_ret = reg.fit(X, y).predict(T)
        ret_list = pd.DataFrame(pred_ret, columns=train_df_known.columns[:12])
        _intersection_arr = test_df_known['intersection_id']
        _tollgate_id_arr = test_df_known['tollgate_id']
        _date_arr = test_df_known['date']
        ret_list.insert(0, 'date', np.array(_date_arr))
        ret_list.insert(0, 'tollgate_id', np.array(_tollgate_id_arr))
        ret_list.insert(0, 'intersection_id', np.array(_intersection_arr))
        actual_df = actual_df.append(to_submit_format_df(test_df_known, flag = False), ignore_index=True)
        pred_df = pred_df.append(to_submit_format_df(ret_list), ignore_index=True)
    write_to_file(actual_df, '10.18-10.24-6:00-8:00.csv')
    write_to_file(pred_df, 'task1_result_Model_2.csv')
