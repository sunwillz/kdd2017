# -*- coding: utf-8 -*-
"""
模型3：对每条A->2,A->3,B->1,B->3,C->1,C->3上的link建立模型,最简单的方法是根据训练数据求出每段link的均值
则一条轨迹的平均时间由组成这条轨迹的link的平均时间加和
"""
import os
import datetime
from os import path
import pandas as pd
import numpy as np

from kdd.metrics import task1_eva_metrics

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


def data_filter(data, start_date, end_date, flag = False):
    """取每天的8:00:00-10:00:00和17:00:00-19:00:00时间段内的数据
    :param data:
    :return:
    """
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    i = datetime.timedelta(days=1)
    data_df_known = pd.DataFrame(columns=data.columns)#已知前两小时
    data_df_pred = pd.DataFrame(columns=data.columns)#要预测的后两小时
    while i <= (end_date - start_date):
        cur_date = (start_date + i).strftime('%Y-%m-%d')
        if flag:
            data_8_10 = data[(data.starting_time >= cur_date+" 08:00:00") & (data.starting_time <= cur_date+" 10:00:00")]
            data_17_19 = data[(data.starting_time >= cur_date+" 17:00:00") & (data.starting_time <= cur_date+" 19:00:00")]
            data_df_pred = data_df_pred.append(data_8_10, ignore_index=True)
            data_df_pred = data_df_pred.append(data_17_19, ignore_index=True)

        else:
            data_6_8 = data[(data.starting_time >= cur_date+" 06:00:00") & (data.starting_time <= cur_date+" 08:00:00")]

            data_15_17 = data[(data.starting_time >= cur_date+" 15:00:00") & (data.starting_time <= cur_date+" 17:00:00")]

            data_df_known = data_df_known.append(data_6_8, ignore_index=True)
            data_df_known = data_df_known.append(data_15_17, ignore_index=True)

        i += datetime.timedelta(days=1)
    return data_df_pred if flag else data_df_known


def get_avg_time(data,weekday,flag = False):
    """
    由指定的星期几,link_id和时间段获取平均时间(ETA)
    :param link_id:
    :param time_window:
    :return:
    """
    if flag:
        time_inteval = [
            ['06:00:00', '06:20:00'], ['06:20:00', '06:40:00'], ['06:40:00', '07:00:00'], ['07:00:00', '07:20:00'],
            ['07:20:00', '07:40:00'], ['07:40:00', '08:00:00'],
            ['15:00:00', '15:20:00'], ['15:20:00', '15:40:00'], ['15:40:00', '16:00:00'], ['16:00:00', '16:20:00'],
            ['16:20:00', '16:40:00'], ['16:40:00', '17:00:00']

        ]
    else:
        time_inteval = [
            ['08:00:00', '08:20:00'], ['08:20:00', '08:40:00'], ['08:40:00', '09:00:00'], ['09:00:00', '09:20:00'],
            ['09:20:00', '09:40:00'], ['09:40:00', '10:00:00'],
            ['17:00:00', '17:20:00'], ['17:20:00', '17:40:00'], ['17:40:00', '18:00:00'], ['18:00:00', '18:20:00'],
            ['18:20:00', '18:40:00'], ['18:40:00', '19:00:00']

        ]
    #未考虑轨迹
    link_time = {}
    for index, row in data.iterrows():
        trajectory = row['travel_seq']
        links = trajectory.split(';')
        start_time_date = links[0].split('#')[1]
        cur_weekday = datetime.datetime.strptime(start_time_date, '%Y-%m-%d %H:%M:%S').weekday()  # 获取是星期几，返回5表示星期六
        #只考虑一周的同一天的数据
        # if (cur_weekday+1) != weekday:
        #     continue
        for link in links:
            _array = link.split('#')
            link_id = _array[0]
            start_time_time = _array[1].split(' ')[1]
            travel_time = _array[2]
            if link_id not in link_time.keys():
                link_time[link_id] = {}
                for window in range(1, 13):
                    link_time[link_id]['travel_time'+str(window)] = []
            for index, item in enumerate(time_inteval):
                if start_time_time>=item[0] and start_time_time<=item[1]:
                    link_time[link_id]['travel_time'+str(index+1)].append(float(travel_time))

    for link_id, time_dic in link_time.iteritems():
        for window, time_list in time_dic.iteritems():
            _list = link_time[link_id][window]
            _len = len(_list)
            _list.sort()
            _list = _list[int(_len*0.1):int(_len*0.9)]
            _list_mean = float(sum(_list))/len(_list)
            link_time[link_id][window] =_list_mean

    return link_time

def validaition():
    """
    本地测试
    :return:
    """
    #本地交叉验证，去训练集最后一个星期作为验证集
    training_file = 'trajectories(table 5)_training.csv'
    train_df = read_training_data(training_file)
    train_df = drop_spec_date(train_df)

    training_data_df = data_filter(train_df, start_date='2016-07-18', end_date='2016-10-17')

    time_inteval_known = [
        ['06:00:00', '06:20:00'], ['06:20:00', '06:40:00'], ['06:40:00', '07:00:00'], ['07:00:00', '07:20:00'],
        ['07:20:00', '07:40:00'], ['07:40:00', '08:00:00'],
        ['15:00:00', '15:20:00'], ['15:20:00', '15:40:00'], ['15:40:00', '16:00:00'], ['16:00:00', '16:20:00'],
        ['16:20:00', '16:40:00'], ['16:40:00', '17:00:00']

    ]
    time_inteval_pred = [
            ['08:00:00', '08:20:00'], ['08:20:00', '08:40:00'], ['08:40:00', '09:00:00'], ['09:00:00', '09:20:00'],
            ['09:20:00', '09:40:00'], ['09:40:00', '10:00:00'],
            ['17:00:00', '17:20:00'], ['17:20:00', '17:40:00'], ['17:40:00', '18:00:00'], ['18:00:00', '18:20:00'],
            ['18:20:00', '18:40:00'], ['18:40:00', '19:00:00']

        ]
    roads = [['A', 2], ['A', 3], ['B', 1], ['B', 3], ['C', 1], ['C', 3]]
    trajectories = [
        [110,123,107,108,120,117],
        [123,107,108,119,114,118,122],
        [105,100,111,103,116,101,121,106,113],
        [105,100,111,103,122],
        [115,102,109,104,112,111,103,116,101,121,106,113],
        [115,102,109,104,112,111,103,122]
    ]
    date_array = ['2016-10-18', '2016-10-19', '2016-10-20', '2016-10-21', '2016-10-22', '2016-10-23', '2016-10-24']
    # trajectories[0] = [110,123,107,108,120,117]#'A'-> 2
    # trajectories[1] = [123,107,108,119,114,118,122]#'A'->3
    # trajectories[2] = [105,100,111,103,116,101,121,106,113]#'B'->1
    # trajectories[3] = [105,100,111,103,122]#'B'->3
    # trajectories[4] = [115,102,109,104,112,111,103,116,101,121,106,113]#'C'->1
    # trajectories[5] = [115,102,109,104,112,111,103,122]#'C'->3

    result_df = pd.DataFrame(columns=['intersection_id', 'tollgate_id', 'time_window', 'avg_travel_time'])
    for idx, trajectory in enumerate(trajectories):
        for day in date_array:
            cur_weekday = datetime.datetime.strptime(day, '%Y-%m-%d').weekday()
            link_time = get_avg_time(training_data_df, weekday=(cur_weekday+1), flag=True)
            for time_window in range(1, 13):
                row = []
                row.append(roads[idx][0])
                row.append(roads[idx][1])
                row.append('['+day+' '+time_inteval_known[time_window-1][0]+','+day+' '+time_inteval_known[time_window-1][1]+')')
                travel_time = 0
                for link_id in trajectory:
                    travel_time += link_time[str(link_id)]['travel_time'+str(time_window)]
                row.append(travel_time)
                result_df = result_df.append(pd.DataFrame(np.array([row]), columns=result_df.columns), ignore_index=True)

    result_df[['tollgate_id']] = result_df[['tollgate_id']].astype(int)
    result_df[['avg_travel_time']] = result_df[['avg_travel_time']].astype(float)

    task1_eva_metrics(result_df.copy(), pd.read_csv('10.18-10.24-6:00-8:00.csv', header=0))


def test():
    """
    针对给定的测试机给出预测结果
    :return: predict result
    """

    training_file = 'trajectories(table 5)_training.csv'
    train_df = read_training_data(training_file)
    train_df = drop_spec_date(train_df)

    training_data_df = data_filter(train_df, start_date='2016-07-18', end_date='2016-10-17', flag=True)
    time_inteval_pred = [
            ['08:00:00', '08:20:00'], ['08:20:00', '08:40:00'], ['08:40:00', '09:00:00'], ['09:00:00', '09:20:00'],
            ['09:20:00', '09:40:00'], ['09:40:00', '10:00:00'],
            ['17:00:00', '17:20:00'], ['17:20:00', '17:40:00'], ['17:40:00', '18:00:00'], ['18:00:00', '18:20:00'],
            ['18:20:00', '18:40:00'], ['18:40:00', '19:00:00']

        ]
    roads = [['A', 2], ['A', 3], ['B', 1], ['B', 3], ['C', 1], ['C', 3]]
    trajectories = [
        [110, 123, 107, 108, 120, 117],
        [123, 107, 108, 119, 114, 118, 122],
        [105, 100, 111, 103, 116, 101, 121, 106, 113],
        [105, 100, 111, 103, 122],
        [115, 102, 109, 104, 112, 111, 103, 116, 101, 121, 106, 113],
        [115, 102, 109, 104, 112, 111, 103, 122]
    ]
    date_array = ['2016-10-18', '2016-10-19', '2016-10-20', '2016-10-21', '2016-10-22', '2016-10-23', '2016-10-24']
    # trajectories[0] = [110,123,107,108,120,117]#'A'-> 2
    # trajectories[1] = [123,107,108,119,114,118,122]#'A'->3
    # trajectories[2] = [105,100,111,103,116,101,121,106,113]#'B'->1
    # trajectories[3] = [105,100,111,103,122]#'B'->3
    # trajectories[4] = [115,102,109,104,112,111,103,116,101,121,106,113]#'C'->1
    # trajectories[5] = [115,102,109,104,112,111,103,122]#'C'->3

    result_df = pd.DataFrame(columns=['intersection_id', 'tollgate_id', 'time_window', 'avg_travel_time'])
    for idx, trajectory in enumerate(trajectories):
        for day in date_array:
            cur_weekday = datetime.datetime.strptime(day, '%Y-%m-%d').weekday()
            link_time = get_avg_time(training_data_df, weekday=(cur_weekday+1))
            for time_window in range(1, 13):
                row = []
                row.append(roads[idx][0])
                row.append(roads[idx][1])
                row.append('['+day+' '+time_inteval_pred[time_window-1][0]+','+day+' '+time_inteval_pred[time_window-1][1]+')')
                travel_time = 0
                for link_id in trajectory:
                    travel_time +=link_time[str(link_id)]['travel_time'+str(time_window)]
                row.append(travel_time)
                result_df = result_df.append(pd.DataFrame(np.array([row]), columns=result_df.columns), ignore_index=True)

    result_df[['tollgate_id']] = result_df[['tollgate_id']].astype(int)
    result_df[['avg_travel_time']] = result_df[['avg_travel_time']].astype(float)
    result_df.to_csv('task1_model_3.submit.csv', index=False)