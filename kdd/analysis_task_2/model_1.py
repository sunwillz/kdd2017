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
"""

from __future__ import division
from __future__ import print_function

from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd

import data as task_2_data
from ..util import parse_freq


def write_prediction_to_csv(file_name, data):
    """
    :param file_name: 保存数据的路径
    :param data: 数据对象, 是`transformed_to_standard_predict`函数的返回
    :return:
    """
    data.to_csv(file_name, index=False)


def predict(online=False, n_neighbors=4, freq="20Min"):
    """返回官方要求的20Min interval数据格式"""
    pred = _predict(online, n_neighbors, freq=freq)
    return transformed_to_standard_data(pred)


def get_test_data_standard():
    """返回官方要求的20Min interval数据格式"""
    offline_test_data = task_2_data.get_offline_test_data()
    test_data_list = get_test_data_list(offline_test_data)
    return transformed_to_standard_data(test_data_list)


def transformed_to_standard_data(data_list):
    """
    :param data_list: predict(online=online, n_neighbors=n_neighbors)这种格式数据
    :return: 标准化预测为官方要求格式
    """
    _columns_8_10 = task_2_data.get_columns_2hours('08:00:00', freq='20Min')[:]
    _columns_8_10.append("10:00:00")
    _columns_17_19 = task_2_data.get_columns_2hours('17:00:00', freq='20Min')[:]   # deep copy
    _columns_17_19.append("19:00:00")

    columns = ['tollgate_id', 'time_window', 'direction', 'volume']
    standard_predict = pd.DataFrame(columns=columns)
    for (_tollgate_id, _direction_id), pred in zip(task_2_data.tollgate_direction_list, data_list):
        for pre, time_seq in [(pred[0], _columns_8_10), (pred[1], _columns_17_19)]:
            window = ["[" + day + " " + beg + "," + day + " " + end + ")" for day in pre.index for beg, end in
                      zip(time_seq[:-1], time_seq[1:])]
            values = pre.values.reshape((-1,))  # 展平成一维
            df = pd.DataFrame({
                columns[0]: _tollgate_id,   # tollgate_id
                columns[1]: window,         # time_window
                columns[2]: _direction_id,  # direction
                columns[3]: values          # volume
            }, columns=columns)
            standard_predict = standard_predict.append(df, ignore_index=True)
    standard_predict[[columns[0], columns[2]]] = standard_predict[
        [columns[0], columns[2]]].astype(np.int64).astype(str)                          # str type
    standard_predict[[columns[3]]] = standard_predict[[columns[3]]].astype(np.int64)    # int type
    return standard_predict


def get_test_data_list(off_line_test_data):
    """返回的数据和predict格式一样, shape也一样"""
    res_data = []
    for filter_data_8_10, filter_data_17_19 in off_line_test_data:      # loop 5
        indexs = np.unique(filter_data_8_10.index.strftime("%Y-%m-%d"))
        filter_data_8_10 = pd.DataFrame(filter_data_8_10.values.reshape(-1, 6), index=indexs,
                                        columns=task_2_data.get_columns_2hours('08:00:00', freq='20Min'))
        filter_data_17_19 = pd.DataFrame(filter_data_17_19.values.reshape(-1, 6), index=indexs,
                                         columns=task_2_data.get_columns_2hours('17:00:00', freq='20Min'))
        res_data.append((filter_data_8_10, filter_data_17_19))
    return res_data


def _predict(online=False, n_neighbors=5, freq="20Min"):
    """
    :return:  返回20Min预测数据, list
    """
    # filter_data固定在20Min过滤， 因为我们只需要在找到邻居的时候使用freq
    filter_data = task_2_data.get_online_filter_data(freq='20Min') if online else task_2_data.get_offline_filter_data(freq='20Min')
    indices = nearest_neighbour(online=online, n_neighbors=n_neighbors, freq=freq)  # 返回最相近的邻居的索引
    res_predict = []
    interval = parse_freq(freq)

    for filter_volumes, indices_filter in zip(filter_data, indices):    # loop 5
        filter_data_8_10, filter_data_17_19 = filter_volumes
        indices_for_8_10, indices_for_17_19 = indices_filter
        indexs = np.unique(filter_data_8_10.index.strftime("%Y-%m-%d"))     # date
        filter_data_8_10 = pd.DataFrame(filter_data_8_10.values.reshape(-1, 6), index=indexs,
                                        columns=task_2_data.get_columns_2hours('08:00:00', freq='20Min'))
        filter_data_17_19 = pd.DataFrame(filter_data_17_19.values.reshape(-1, 6), index=indexs,
                                         columns=task_2_data.get_columns_2hours('17:00:00', freq='20Min'))

        throught_day_list = []
        for _filter, _indices in [(filter_data_8_10, indices_for_8_10), (filter_data_17_19, indices_for_17_19)]:
            pred = pd.DataFrame(columns=_filter.columns, dtype=np.int64)
            _indice, days = _indices
            for i, a_indice in enumerate(_indice):
                one_day_predict_data = _filter.loc[a_indice, :].mean().round().astype(np.int64)     # 转化为整数
                pred.loc[days[i]] = one_day_predict_data    # 将数据添加到dataframe里面去
            throught_day_list.append(pred)
        res_predict.append(tuple(throught_day_list))
    return res_predict


def nearest_neighbour(online=False, n_neighbors=5, freq="20Min"):
    """
    :param online: bool. 是否是线上
    :param n_neighbors: int. 邻居数目
    :param freq:
    :return:  list.
        解释:
        len(return)=5, 对应着tollgate_direction_list=[(1, 0), (1, 1), (2, 0), (3, 0), (3, 1)]
        return.shape = (5,2,2). 对于第三维的数据,
        return[0][0] = (indices, days)  and for every (indices, days)
        indices.shape = (len(fit_volume_6_8)/6, n_neighbors), days.shape = (len(fit_volume_6_8)/6, )
        indices[i] 为 days[i]这个时间点的数据最相近的日期
    """
    if online:
        train_data = task_2_data.get_online_train_data(freq=freq)
        fit_data = task_2_data.get_online_fit_data(freq=freq)
    else:
        train_data = task_2_data.get_offline_train_data(freq=freq)
        fit_data = task_2_data.get_offline_fit_data(freq=freq)

    knn_param = {       # NearestNeighbors 类实例化参数
        "n_neighbors": n_neighbors,
        'algorithm': 'auto',
        'p': 1,
        # 'metric': lambda pred, act: sum([abs((a-p)/a) for a, p in zip(act, pred)]),  # 指标来自kddcup2017 Metrics task2
    }
    interval = parse_freq(freq)
    indices = []    # result
    for fit_volumes, train_volumes in zip(fit_data, train_data):        # loop 5
        fit_volume_6_8, fit_volume_15_17 = fit_volumes
        train_volume_6_8, train_volume_15_17 = train_volumes

        fit_days_list = np.unique(fit_volume_6_8.index.strftime("%Y-%m-%d"))
        fit_volume_6_8 = fit_volume_6_8.values.reshape(-1, interval)       # shape = (7, interval)
        fit_volume_15_17 = fit_volume_15_17.values.reshape(-1, interval)

        train_days_list = np.unique(train_volume_6_8.index.strftime("%Y-%m-%d"))
        train_volume_6_8 = train_volume_6_8.values.reshape(-1, interval)       # shape = (22, interval)
        train_volume_15_17 = train_volume_15_17.values.reshape(-1, interval)

        nbrs_6_8 = NearestNeighbors(**knn_param).fit(train_volume_6_8)
        # distance 是得到最近的n_neighbours的距离. shape=(len(fit_volume_6_8), n_neighbors)
        # indices 是最近的n_neighbours的索引  shape=(len(fit_volume_6_8), n_neighbors)
        indices_6_8 = nbrs_6_8.kneighbors(fit_volume_6_8, return_distance=False)
        # indices_6_8 = _transform_indices_to_date(indices_6_8, train_days_list)
        indices_6_8 = train_days_list[indices_6_8]      # 将indices转化成日期

        nbrs_15_17 = NearestNeighbors(**knn_param).fit(train_volume_15_17)
        # distance 是得到最近的n_neighbours的距离. indices 是最近的n_neighbours的索引
        indices_15_17 = nbrs_15_17.kneighbors(fit_volume_15_17, return_distance=False)
        # indices_15_17 = _transform_indices_to_date(indices_15_17, train_days_list)
        indices_15_17 = train_days_list[indices_15_17]  # 将indices转化成日期

        indices.append(((indices_6_8, fit_days_list), (indices_15_17, fit_days_list)))
    return indices

