# -*- coding: utf-8 -*-

"""
数据:
    线下:
        训练数据: 2016.09.19 ~ 2016.10.10       # 3个星期
        测试数据: 2016.10.11 ~ 2016.10.17       # 1个星期
            给定: 06:00 - 08:00 and 15:00 - 17:00
            预测: 08:00 - 10:00 and 17:00 - 19:00, at 20-minute intervals.
    线上:
        训练数据: 2016.09.19 ~ 2016.10.17       # 4个星期
        测试数据: 2016.10.18 ~ 2016.10.24       # 1个星期
            给定: 06:00 - 08:00 and 15:00 - 17:00
            预测: 08:00 - 10:00 and 17:00 - 19:00, at 20-minute intervals.
模型:
    1. 根据训练数据[06:00 - 08:00 and 15:00 - 17:00] 然后去匹配每一条测试数据的[06:00 - 08:00 and 15:00 - 17:00],
    2. 然后找到和测试数据最相近的k条数据, 认为这些日子的路况和测试数据路况最相近
    3. 用这些数据的[08:00 - 10:00 and 17:00 - 19:00, at 20-minute intervals.]
        去预测测试数据的[08:00 - 10:00 and 17:00 - 19:00, at 20-minute intervals.]
该模型和model_1模型的区别在于这个模型是使用 KNeighborsRegressor
"""

from __future__ import division
from __future__ import print_function

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

import data as task_2_data
from ..util import parse_freq


def predict(online=False, n_neighbors=4, freq="20Min", drop_dates=None):     # 对于22天包括十一, 比较好的参数是(7, 20Min), (9, 30Min)
    """
    返回官方要求的20Min interval数据格式
    drop_dates是去掉一些日期，详情看`database.get_volume_by_time`的参数
    """
    pred = _predict(online, n_neighbors, freq=freq, drop_dates=drop_dates)
    return task_2_data.transformed_to_standard_data(pred)


def _predict(online=False, n_neighbors=5, freq="20Min", drop_dates=None):
    """
    :return:  返回20Min预测数据, list
    """
    preds, pred_days = nearest_neighbour(online=online, n_neighbors=n_neighbors, freq=freq, drop_dates=drop_dates)  # 返回最相近的邻居的索引
    res_predict = []
    for pred in preds:    # loop 5
        pred_for_8_10, pred_for_17_19 = pred
        pred_for_8_10 = pd.DataFrame(pred_for_8_10, index=pred_days,
                                     columns=task_2_data.get_columns_2hours('08:00:00', freq='20Min'))
        pred_for_17_19 = pd.DataFrame(pred_for_17_19, index=pred_days,
                                      columns=task_2_data.get_columns_2hours('17:00:00', freq='20Min'))
        res_predict.append((pred_for_8_10, pred_for_17_19))
    return res_predict


def nearest_neighbour(online=False, n_neighbors=5, freq="20Min", drop_dates=None):
    """
    :param online: bool. 是否是线上
    :param n_neighbors: int. 邻居数目
    :param freq:
    :param drop_dates: list
    :return:  pred_list, days.
        解释:
        pred_list. len(pred_list)=5, 对应着tollgate_direction_list=[(1, 0), (1, 1), (2, 0), (3, 0), (3, 1)]
        pred_list.shape = (5,2).
        pred_list[0][0] = pred
        days为预测的日期
    """
    if online:
        train_data = task_2_data.get_online_train_data(freq=freq, drop_dates=drop_dates)   # train_x
        fit_data = task_2_data.get_online_fit_data(freq=freq)       # test_x
        train_y = task_2_data.get_online_filter_data(freq='20Min', drop_dates=drop_dates)  # train_y
    else:
        train_data = task_2_data.get_offline_train_data(freq=freq, drop_dates=drop_dates)  # train_x
        fit_data = task_2_data.get_offline_fit_data(freq=freq)      # test_x
        train_y = task_2_data.get_offline_filter_data(freq='20Min', drop_dates=drop_dates) # train_y

    knn_param = {       # NearestNeighbors 类实例化参数
        "n_neighbors": n_neighbors,
        'algorithm': 'auto',
        'p': 1,
        'weights': 'distance',
        # 'metric': lambda pred, act: sum([abs((a-p)/a) for a, p in zip(act, pred)]),  # 指标来自kddcup2017 Metrics task2
    }
    interval = parse_freq(freq)
    pred = []    # result
    for fit_volumes, train_volumes, y_volumes in zip(fit_data, train_data, train_y):        # loop 5
        day_predict = []
        for fit_volume, train_volume, y_volume in zip(fit_volumes, train_volumes, y_volumes):

            # train_days_list = np.unique(train_volume.index.strftime("%Y-%m-%d"))   # 调试

            fit_days_list = np.unique(fit_volume.index.strftime("%Y-%m-%d"))    # test_days
            train_volume = train_volume.values.reshape(-1, interval)  # shape = (22, interval)
            fit_volume = fit_volume.values.reshape(-1, interval)       # shape = (7, interval)
            y_volume = y_volume.values.reshape(-1, 6)
            nbrs = KNeighborsRegressor(**knn_param).fit(train_volume, y_volume)

            # 调试
            # distance, indexs = nbrs.kneighbors(fit_volume)
            # print(train_days_list[indexs])
            # print(distance)
            # print(fit_volume)
            # return

            volume_predict = nbrs.predict(fit_volume)
            day_predict.append(volume_predict)
        pred.append(tuple(day_predict))
    return pred, fit_days_list  # fit_day_list 一直是固定的,就是要预测的那几天的数据

# (3,1)和(1,1) 相似度很高，可以拿这两个互相补充数据预测