# -*- coding: utf-8 -*-

"""
模型一：历史平均模型：用训练集里面与测试集最相近的10天的交通情况预测未来的情况,未区分周末和非周末
数据：
    训练数据：2016.07.19-2016.10.17,三个月全天24小时的车辆轨迹信息
    预测数据：2016.10.18-2016.10.24,[6：00-8:00][15:00-17:00]一个星期的车辆轨迹信息

划分测试集：
    取训练数据最后一个星期作为测试集
    训练集：2016.07.19-2016.10.10,[6:00-8:00][15:00-17:00]和[8:00-10:00][17:00-19:00]
    测试集：2016.10.11-2016.10.17,已知[6:00-8:00][15:00-17:00]预测[8:00-10:00][17:00-19:00]
"""
import os
from os import path
import datetime
import pandas as pd
import numpy as np
import re
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor

from kdd.util import PROJECT_PATH

from kdd.metrics import task1_eva_metrics

# PROJECT_PATH = os.path.dirname(os.path.abspath(__name__))
dir_path = path.join(PROJECT_PATH, "datasets/dataSets/")
train_data_path = path.join(dir_path, "training/")
test_data_path = path.join(dir_path, "testing_phase1/")

def read_training_data(filename):
    """
    读取训练数据
    :param filename:
    :return:
    """
    path = train_data_path+filename
    # colume_names = ["intersection_id", "tollgate_id", "vehicle_id", "starting_time", "travel_seq", "travel_time"]
    data_df = pd.read_csv(path, header=0)
    return data_df
def read_testing_data(filename):
    """
    读取测试数据
    :param filename:
    :return:
    """
    path = test_data_path + filename
    # colume_names = ["intersection_id", "tollgate_id", "vehicle_id", "starting_time", "travel_seq", "travel_time"]
    data_df = pd.read_csv(path, header=0)
    return data_df
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

def get_data_by_date(data, start_date, end_date):
    """
    三个月不含最后一个星期的数据（2016-07-19,2016-10-10）
    :param data:
    :param start_date: 输入数据需要提前一天，且是string类型
    :param end_date: 输出数据为当天日期，string类型
    :return:
    """
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    i = datetime.timedelta(days=1)
    data_df_known = pd.DataFrame(columns=data.columns)#已知前两小时
    data_df_pred = pd.DataFrame(columns=data.columns)#要预测的后两小时
    while i <= (end_date - start_date):
        cur_date = (start_date + i).strftime('%Y-%m-%d')
        data_6_8 = data[(data.starting_time >= cur_date+" 06:00:00") & (data.starting_time <= cur_date+" 08:00:00")]
        data_8_10 = data[(data.starting_time >= cur_date+" 08:00:00") & (data.starting_time <= cur_date+" 10:00:00")]
        data_15_17 = data[(data.starting_time >= cur_date+" 15:00:00") & (data.starting_time <= cur_date+" 17:00:00")]
        data_17_19 = data[(data.starting_time >= cur_date+" 17:00:00") & (data.starting_time <= cur_date+" 19:00:00")]
        data_df_known = data_df_known.append(data_6_8, ignore_index=True)
        data_df_known = data_df_known.append(data_15_17, ignore_index=True)
        data_df_pred = data_df_pred.append(data_8_10, ignore_index=True)
        data_df_pred = data_df_pred.append(data_17_19, ignore_index=True)
        i += datetime.timedelta(days=1)

    return data_df_known, data_df_pred

def set_missing_data(df):
    """
    处理缺失值，用均值代替
    :param df:
    :return:
    """
    return df.fillna(df.mean())

def form_training_data(data, start_date, end_date):
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


def to_submit_format_df(df):
    """
    转成提交文件的格式
    :param actual_df:
    :param pred_df:
    :return:
    """
    time_inteval_pred = [
        ['08:00:00', '08:20:00'], ['08:20:00', '08:40:00'], ['08:40:00', '09:00:00'], ['09:00:00', '09:20:00'], ['09:20:00', '09:40:00'], ['09:40:00', '10:00:00'],
        ['17:00:00', '17:20:00'], ['17:20:00', '17:40:00'], ['17:40:00', '18:00:00'], ['18:00:00', '18:20:00'], ['18:20:00', '18:40:00'], ['18:40:00', '19:00:00']

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
                time_window = "["+_date+" "+time_inteval_pred[idx-1][0]+","+_date+" "+time_inteval_pred[idx-1][1]+")"
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


def ETA_predict():
    """
    基于相似度预测
    例如要预测2016.10.11（2016.10.18）08:00:00-10:00:00和17:00:00-19:00:00的ETA
    可以从训练集中找出最相似的n天数据的均值作为预测值
    :param train_data: 训练数据
    :return: 预测结果
    """
    #构造特征向量，按每20分钟分割数据

    #本地划分训练集和测试集用于本地测试
    training_file = 'trajectories(table 5)_training.csv'
    train_df = read_training_data(training_file)
    train_df = drop_spec_date(train_df)
    # 读取测试数据
    # test_df = read_testing_data(testing_file)

    train_df_known, train_df_pred = form_training_data(train_df, start_date='2016-07-18', end_date='2016-10-10')
    test_df_known, test_df_pred = form_training_data(train_df, start_date='2016-10-10', end_date='2016-10-17')

    #由于训练集中缺少2016.08.26 B->1 8:00-10:00和17:00-19:00的数据，因此在train_df_knownz中需要去除当天B->1的数据，以免噪声干扰

    train_df_known_above = train_df_known.loc[:189]
    train_df_known_below = train_df_known.loc[191:]
    train_df_known = train_df_known_above.append(train_df_known_below, ignore_index=True)

    #对给定的测试集进行预测
    # train_df_known, train_df_pred = form_training_data(train_df, start_date='2016-07-18', end_date='2016-10-17')
    # test_df_known = form_testing_data(test_df, start_date='2016-10-17', end_date='2016-10-24')
    #中间的数据保存到文件，方便以后使用
    # test_df_known.to_csv('test_data_format_time.csv', index=False)
    #设定最近邻居为10
    knn_parm={
        'n_neighbors': 10,
        'algorithm': 'auto',
        'metric': 'manhattan',
    }

    neigh = NearestNeighbors(**knn_parm)
    train_df = train_df_known.drop(["intersection_id", "tollgate_id", "date"], axis=1)
    test_df = test_df_known.drop(["intersection_id", "tollgate_id", "date"], axis=1)
    neigh.fit(train_df)
    #得到最相近的10个邻居下标和距离，是个二维列表
    distances, indices = neigh.kneighbors(test_df, return_distance=True)
    # 邻居的权重
    # weights = [0.3, 0.25, 0.14, 0.1, 0.05, 0.04, 0.03, 0.03, 0.03, 0.03]
    weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    # np.savetxt('neighber.txt', neighbors, delimiter=',')
    ret_list = pd.DataFrame()
    for row in indices:
        df = pd.DataFrame(np.zeros((1, 12)), columns=train_df_known.columns[:12])
        for index, item in enumerate(row):
            intersection_id = train_df_known.iloc[item]['intersection_id']
            tollgate_id = train_df_known.iloc[item]['tollgate_id']
            _date =train_df_known.iloc[item]['date']
            sim_day = train_df_pred[(train_df_pred.intersection_id == intersection_id) & (train_df_pred.tollgate_id == tollgate_id) & (train_df_pred.date == _date)]
            tp_1 = sim_day.iloc[:, :12].reset_index()
            del tp_1['index']
            tp_2 = df.iloc[:, :12]
            df = tp_1*weights[index]+tp_2
            # df = df.append(tp*weights[index], ignore_index=True)
        ret_list = ret_list.append(df, ignore_index=True)
    # del ret_list['tollgate_id']
    ret_list = ret_list.fillna(ret_list.mean())
    for idx in range(12, 0, -1):
        col = ret_list.pop('travel_time'+str(idx))
        ret_list.insert(0, 'travel_time'+str(idx), col)
    #在本地测试的时候去掉该注释
    actual_df = test_df_pred
    _intersection_arr = test_df_known['intersection_id']
    _tollgate_id_arr = test_df_known['tollgate_id']
    _date_arr = test_df_known['date']
    ret_list.insert(0, 'date', np.array(_date_arr))
    ret_list.insert(0, 'tollgate_id', np.array(_tollgate_id_arr))
    ret_list.insert(0, 'intersection_id', np.array(_intersection_arr))

    actual_df = to_submit_format_df(actual_df)
    pred_df = to_submit_format_df(ret_list)

    # write_to_file(pred_df, 'task1_result.csv')

    return task1_eva_metrics(pred_df.copy(), actual_df.copy())


training_file = 'trajectories(table 5)_training.csv'
testing_file = 'trajectories(table 5)_test1.csv'

# 读取训练数据
train_df = read_training_data(training_file)

# 读取测试数据
test_df = read_testing_data(testing_file)

print "MAPE: ", ETA_predict(train_df, test_df)

