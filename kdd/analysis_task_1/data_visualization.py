# -*- coding: utf-8 -*-
"""
数据可视化类,用于将训练数据绘成图标，方便进行数据分析
"""

import os
import datetime
from os import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
PROJECT_PATH = os.path.dirname(os.path.abspath(__name__))
dir_path = path.join(PROJECT_PATH, "datasets/dataSets/")
train_data_path = path.join(dir_path, "training/")
test_data_path = path.join(dir_path, "testing_phase1/")
training_file = 'trajectories(table 5)_training.csv'


def read_training_data(filename):
    filepath = train_data_path+filename
    return pd.read_csv(filepath, header=0)


def show_travel_time_in_window(data, intersection_id, tollgate_id):

    """
    根据给定的参数返回数据趋势图
    :param intersection_id:
    :param tollagte_id:
    :param time_window:
    :return:
    """

    time_inteval = [
        ['06:00:00', '06:20:00'], ['06:20:00', '06:40:00'], ['06:40:00', '07:00:00'], ['07:00:00', '07:20:00'],
        ['07:20:00', '07:40:00'], ['07:40:00', '08:00:00'],
        ['15:00:00', '15:20:00'], ['15:20:00', '15:40:00'], ['15:40:00', '16:00:00'], ['16:00:00', '16:20:00'],
        ['16:20:00', '16:40:00'], ['16:40:00', '17:00:00']

    ]
    # time_window = time_inteval[time_window_index]
    start_date = datetime.datetime.strptime('2016-07-18', '%Y-%m-%d')
    end_date = datetime.datetime.strptime('2016-10-17', '%Y-%m-%d')
    for index in range(len(time_inteval)):
        time_window = time_inteval[index]
        i = datetime.timedelta(days=1)
        data_df = pd.DataFrame(columns=['intersection_id', 'tollgate_id', 'date', 'travel_time'])
        while i <= (end_date - start_date):
            cur_date = (start_date + i).strftime('%Y-%m-%d')
            _data = data[(data.intersection_id == intersection_id) & (data.tollgate_id == tollgate_id) & (data.starting_time >= cur_date + " " +time_window[0]) & (data.starting_time <= cur_date + " " +time_window[1])]
            temp_data_df = pd.DataFrame([[intersection_id, tollgate_id, cur_date, _data['travel_time'].mean()]], columns=['intersection_id', 'tollgate_id', 'date', 'travel_time'])
            data_df = data_df.append(temp_data_df, ignore_index=True)
            i += datetime.timedelta(days=1)
        data_df = data_df.fillna(data_df.mean())
        ret_array = data_df['travel_time']
        # plt.subplot(12, 1, int(index+1))
        plt.plot(pd.DatetimeIndex(data_df['date']), ret_array, label=str(time_inteval[index]))
    plt.legend(loc='right upper')
    plt.grid(True)
    plt.xlabel('date')
    plt.ylabel('avg travel time ')
    plt.show()