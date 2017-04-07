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
该模型和model_1模型的区别在于这个模型是使用 KNeighborsRegressor. 同时在预测中提供了drop_days的功能去除掉异常日期。
"""

from __future__ import division
from __future__ import print_function

from datetime import datetime, timedelta

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

from ..data_preprocess.data_read import database
import data as task_2_data
from ..util import parse_freq


def predict(online=False, n_neighbors=7, freq="20Min", drop_dates=task_2_data.abnormal_days, threshold=1):     # 对于22天包括十一, 比较好的参数是(7, 20Min), (9, 30Min)
    """
    返回官方要求的20Min interval数据格式
    drop_dates是去掉一些日期. 类型为一个（5,2)的列表. 每个元素详情看`database.get_volume_by_time`的参数, 或者data.abnormal_days
    """
    pred = _predict(online, n_neighbors, freq=freq, drop_dates=drop_dates, threshold=threshold)
    return task_2_data.transformed_to_standard_data(pred)


def _predict(online=False, n_neighbors=5, freq="20Min", drop_dates=None, threshold=0.25):
    """
    :return:  返回20Min预测数据, list
    """
    preds, pred_days, n_indexes = nearest_neighbour(online=online, n_neighbors=n_neighbors, freq=freq, drop_dates=drop_dates)  # 返回最相近的邻居的索引
    res_predict = []
    for pred, indexes, toll_direction in zip(preds, n_indexes, task_2_data.tollgate_direction_list):    # loop 5
        pred_for_8_10, pred_for_17_19 = pred
        indexes_for_8_10, indexes_for_17_19 = indexes
        pred_for_8_10 = _deal_abnormal_predict_days(pred_for_8_10, indexes_for_8_10, pred_days, toll_direction,
                                                    is_morning=True, threshold=threshold, online=online)
        pred_for_17_19 = _deal_abnormal_predict_days(pred_for_17_19, indexes_for_17_19, pred_days, toll_direction,
                                                     is_morning=False, threshold=threshold, online=online)

        pred_for_8_10 = pd.DataFrame(pred_for_8_10, index=pred_days,
                                     columns=task_2_data.get_columns_2hours('08:00:00', freq='20Min'))
        pred_for_17_19 = pd.DataFrame(pred_for_17_19, index=pred_days,
                                      columns=task_2_data.get_columns_2hours('17:00:00', freq='20Min'))
        res_predict.append((pred_for_8_10, pred_for_17_19))
    return res_predict


def _deal_abnormal_predict_days(pred, indexes, pred_days, toll_direction, is_morning, threshold, online):
    """
    :param pred:  根据knn regression 预测出来的数据
    :param indexes:  预测出来pred的n个日期
    :param pred_days: 要预测的那些日期， 7天
    :param toll_direction:
    :param is_morning:  是不是上午
    :param threshold:  阈值用来判断是不是异常点. 如threshold=0.25表示邻居里面有0.25为异常点（10-1）里面的日期即为异常预测. 超过1的时候即不考虑异常点
    :return: 处理过异常预测日期的pred
    """
    shi_yi = set(database.holiday)
    for _i, (n_neigh, day) in enumerate(zip(indexes, pred_days)):
        if _is_abnormal_predict_days(shi_yi, n_neigh, threshold):
            pred = _replace_predict(_i, day, pred, toll_direction, is_morning, online)

    return pred


def _replace_predict(_i, day, pred, toll_direction, is_morning, online):
    """
    :param _i: [0-6] 为位置索引
    :param day: 需要做rule处理的日期, 这个日期为预测的日期
    :param pred: 之前的knnregressor做的回归预测
    :param toll_direction:  tollgate, direction对
    :param is_morning: bool, 是不是上午
    :param online: bool, 是否在线
    :return: 做了处理之后的预测数据
    """
    # print(day, toll_direction, 'is_morning='+str(is_morning))
    _day = datetime.strptime(day, "%Y-%m-%d")
    yesterday = (_day + timedelta(days=-1)).strftime("%Y-%m-%d") if not online else '2016-10-17'
    normal_days = np.append(pd.date_range('2016-09-19', '2016-09-29').strftime("%Y-%m-%d")
                            , pd.date_range('2016-10-08', yesterday).strftime("%Y-%m-%d"))
    if is_morning:
        start_time, end_time = "08:00:00", "09:59:59"
    else:
        start_time, end_time = "17:00:00", "18:59:59"
    scaler = _get_scaler(normal_days, day, toll_direction, is_morning, online)
    history_week_x_volume = [database.get_volume_by_time(toll_direction[0], toll_direction[1], start_time=start_time,
                                                         end_time=end_time, start_date=d, end_date=d)
                             for d in normal_days]
    a = pd.Series(np.zeros(len(history_week_x_volume[0])), index=history_week_x_volume[0].index)
    for volume in history_week_x_volume:
        a += volume
    a = scaler * (a/len(history_week_x_volume))
    for i in range(a.shape[0]):
        pred[_i][i] = a[i]

    return pred


def _get_scaler(history_week_x, day, toll_direction, is_morning, online):
    """得到缩放比例"""
    if is_morning:
        start_time, end_time = "06:00:00", "07:59:59"
    else:
        start_time, end_time = "15:00:00", "16:59:59"
    history_week_x_volume = [database.get_volume_by_time(toll_direction[0], toll_direction[1], start_time=start_time,
                                                         end_time=end_time, start_date=d, end_date=d)
                             for d in history_week_x]
    a = pd.Series(np.zeros(len(history_week_x_volume[0])), index=history_week_x_volume[0].index)
    for volume in history_week_x_volume:
        a += volume
    a = a/len(history_week_x_volume)
    # print("异常时间段", day,toll_direction, start_time, end_time)
    is_test_data = True if online else False
    day_volume = database.get_volume_by_time(toll_direction[0], direction=toll_direction[1], start_time=start_time,
                                             end_time=end_time, start_date=day, end_date=day, test_data=is_test_data)
    return day_volume.sum()/a.sum()


def _is_abnormal_predict_days(shi_yi, n_neigh, threshold):
    """判断几个邻居是不是异常点阈值之上"""
    count = 0
    for index in n_neigh:
        if index in shi_yi: count += 1
    return count >= len(n_neigh) * threshold


def nearest_neighbour(online=False, n_neighbors=5, freq="20Min", drop_dates=None):
    """
    :param online: bool. 是否是线上
    :param n_neighbors: int. 邻居数目
    :param freq:
    :param drop_dates: drop_dates是去掉一些日期. 类型为一个（5,2)的列表. 每个元素为一个list, 详情看`database.get_volume_by_time`的参数
    :return:  pred_list , fit_days_list, n_nearest_index.
        解释:
        pred_list. len(pred_list)=5, 对应着tollgate_direction_list=[(1, 0), (1, 1), (2, 0), (3, 0), (3, 1)]
        pred_list.shape = (5,2).
        pred_list[0][0] = pred  pred为流量预测值
        days为预测的日期
    """
    if online:
        train_data = task_2_data.get_online_train_data(freq=freq, drop_dates=drop_dates)   # train_x
        fit_data = task_2_data.get_online_fit_data(freq=freq)                              # test_x
        train_y = task_2_data.get_online_filter_data(freq='20Min', drop_dates=drop_dates)  # train_y
    else:
        train_data = task_2_data.get_offline_train_data(freq=freq, drop_dates=drop_dates)  # train_x
        fit_data = task_2_data.get_offline_fit_data(freq=freq, drop_dates=[([], []), ([], []), ([], []), ([], []), ([], [])])                             # test_x
        train_y = task_2_data.get_offline_filter_data(freq='20Min', drop_dates=drop_dates) # train_y

    knn_param = {       # NearestNeighbors 类实例化参数
        "n_neighbors": n_neighbors,
        'algorithm': 'auto',
        'p': 1,
        'weights': 'distance',
        # 'metric': lambda pred, act: sum([abs((a-p)/a) for a, p in zip(act, pred)]),  # 指标来自kddcup2017 Metrics task2
    }
    n_nearest_index = []  # 用来保存index, shape=(5,2)
    interval = parse_freq(freq)
    pred = []    # result
    # i = 0
    for fit_volumes, train_volumes, y_volumes in zip(fit_data, train_data, train_y):        # loop 5
        day_predict = []
        # j = 0
        n_n_list = []
        for fit_volume, train_volume, y_volume in zip(fit_volumes, train_volumes, y_volumes):
            scaler = MinMaxScaler()

            train_days_list = np.unique(train_volume.index.strftime("%Y-%m-%d"))   # 调试

            fit_days_list = np.unique(fit_volume.index.strftime("%Y-%m-%d"))    # test_days
            train_volume = train_volume.values.reshape(-1, interval)  # shape = (22, interval)
            # 增加median行
            train_volume = scaler.fit_transform(train_volume)
            train_volume = pd.DataFrame(train_volume, index=train_days_list)
            # train_volume['mean'] = train_volume.mean(axis=1).values
            # train_volume['median'] = train_volume.median(axis=1).values
            weights = [1.0, 1.02, 1.04, 1.14, 1.25, 1.35]
            for i, w in enumerate(weights):
                train_volume[train_volume.columns[i]] = train_volume[train_volume.columns[i]].values*w
            # train_volume[train_volume.columns[5]] = train_volume[train_volume.columns[5]].values*1.1

            fit_volume = fit_volume.values.reshape(-1, interval)       # shape = (7, interval)
            # 增加median行
            fit_volume = scaler.transform(fit_volume)
            fit_volume = pd.DataFrame(fit_volume, index=fit_days_list)
            # fit_volume['mean'] = fit_volume.mean(axis=1).values
            # fit_volume['median'] = fit_volume.median(axis=1).values
            for i, w in enumerate(weights):
                fit_volume[fit_volume.columns[i]] = fit_volume[fit_volume.columns[i]].values*w
            # fit_volume[fit_volume.columns[5]] = fit_volume[fit_volume.columns[5]].values * 1.1

            y_volume = y_volume.values.reshape(-1, 6)
            nbrs = KNeighborsRegressor(**knn_param).fit(train_volume, y_volume)

            # 调试
            distance, indexs = nbrs.kneighbors(fit_volume)
            # if i==2 and j==0:
            #     print(train_days_list[indexs])
            # print(distance)
            # print(fit_volume)
            # return

            volume_predict = nbrs.predict(fit_volume)
            n_n_list.append(train_days_list[indexs])
            day_predict.append(volume_predict)
            # j=j+1
        # i=i+1
        n_nearest_index.append(tuple(n_n_list))
        pred.append(tuple(day_predict))
    return pred, fit_days_list, n_nearest_index  # fit_day_list 一直是固定的,就是要预测的那几天的数据
    # (3,1)和(1,1) 相似度很高，可以拿这两个互相补充数据预测
