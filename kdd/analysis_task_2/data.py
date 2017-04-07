# -*- coding: utf-8 -*-

"""

该模块将task2的数据进一步封装

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
"""
import pandas as pd
import numpy as np

from ..data_preprocess import database
from ..util import parse_freq

tollgate_direction_list = [(1, 0), (1, 1), (2, 0), (3, 0), (3, 1)]


# 数据集.  volume_6_8_1_0, 表示6:00-8:00, tollgate=1, direction=0的流量数据
# [(volume_6_8_1_0, volume_15_17_1_0),
#  (volume_6_8_1_1, volume_15_17_1_1),
#  (volume_6_8_2_0, volume_15_17_2_0),
#  (volume_6_8_3_0, volume_15_17_3_0),
#  (volume_6_8_3_1, volume_15_17_3_1)]

offline_train_or_filter_date = ("2016-09-19", "2016-10-10")
offline_fit_or_test_date = ("2016-10-11", "2016-10-17")
online_train_or_filter_date = ("2016-09-19", "2016-10-17")
online_fit_date = ("2016-10-18", "2016-10-24")

_sw_index = ["08:00:00", "08:20:00", "08:40:00", "09:00:00", "09:20:00", "09:40:00"]
_xw_index = ["17:00:00", "17:20:00", "17:40:00", "18:00:00", "18:20:00", "18:40:00"]

# 该常量用于model_1.predict, model_3.predict 的drop_days参数,
abnormal_days = [(['2016-10-01', '2016-09-30'], ['2016-10-01', '2016-09-30']),   # (1,0)
                 (['2016-09-30'], ['2016-09-30']),                           # (1,1)
                 (['2016-09-28', '2016-09-30'], ['2016-09-28', '2016-09-30']),   # (2,0)
                 (['2016-09-30'], ['2016-09-30']),   # (3,0)
                 # (['2016-09-20', '2016-09-30'], ['2016-09-30'])    # (3,1)
                 (['2016-09-30'], ['2016-09-30'])    # (3,1)
                 ]

no_drop_days = [([], []),
                ([], []),
                ([], []),
                ([], []),
                ([], []),
                ]


def get_columns_2hours(start, freq='20Min'):
    """
    :return 返回以freq为频率，start为开始时间的2hours的字符串时间列表
    """
    from pandas import TimedeltaIndex
    return TimedeltaIndex(freq=freq, start=start, periods=parse_freq(freq)).format()    # '08:20:00' 格式


# 线下拟合数据集, 用这些数据集找到和测试数据集最相近的日期
# shape [5, 2, 7*2*3]  2016.10.11 ~ 2016.10.17
# start_time='06:00:00', end_time='07:59:59'
# start_time='15:00:00', end_time='16:59:59'
def get_offline_fit_data(freq='20Min', offline_fit_date=offline_fit_or_test_date, drop_dates=None):
    offline_fit_data = [(database.get_volume_by_time(tollgate_id, direction_id, start_time='06:00:00', end_time='07:59:59',
                                                     sumed_in_one_day=False, freq=freq, drop_dates=sw,
                                                     start_date=offline_fit_date[0], end_date=offline_fit_date[-1]),
                         database.get_volume_by_time(tollgate_id, direction_id, start_time='15:00:00', end_time='16:59:59',
                                                     sumed_in_one_day=False, freq=freq, drop_dates=xw,
                                                     start_date=offline_fit_date[0], end_date=offline_fit_date[-1]))
                        for (tollgate_id, direction_id), (sw, xw) in zip(tollgate_direction_list, drop_dates)]
    return offline_fit_data


# 线下测试数据集
# shape [5, 2, 7*2*3]  2016.10.11 ~ 2016.10.17
# start_time='08:00:00', end_time='09:59:59'
# start_time='17:00:00', end_time='18:59:59'
def get_offline_test_data(freq="20Min", offline_test_date=offline_fit_or_test_date, drop_dates=None):
    offline_test_data = [(database.get_volume_by_time(tollgate_id, direction_id, start_time='08:00:00', end_time='09:59:59',
                                                      sumed_in_one_day=False, freq=freq,
                                                      start_date=offline_test_date[0], end_date=offline_test_date[-1]),
                          database.get_volume_by_time(tollgate_id, direction_id, start_time='17:00:00', end_time='18:59:59',
                                                      sumed_in_one_day=False, freq=freq,
                                                      start_date=offline_test_date[0], end_date=offline_test_date[-1]))
                         # for (tollgate_id, direction_id), (sw, xw) in zip(tollgate_direction_list, drop_dates)]
                         for tollgate_id, direction_id in tollgate_direction_list]
    return offline_test_data


# 线下训练数据集,  2016.09.19 ~ 2016.10.10
def get_offline_train_data(freq="20Min", offline_train_date=offline_train_or_filter_date, drop_dates=None):
    offline_train_data = [(database.get_volume_by_time(tollgate_id, direction_id, start_time='06:00:00', end_time='07:59:59',
                                                       sumed_in_one_day=False, freq=freq, drop_dates=sw,
                                                       start_date=offline_train_date[0], end_date=offline_train_date[-1]),
                          database.get_volume_by_time(tollgate_id, direction_id, start_time='15:00:00', end_time='16:59:59',
                                                      sumed_in_one_day=False, freq=freq, drop_dates=xw,
                                                      start_date=offline_train_date[0], end_date=offline_train_date[-1]))
                          for (tollgate_id, direction_id), (sw, xw) in zip(tollgate_direction_list, drop_dates)]
    return offline_train_data


# 线下过滤数据集,  2016.09.19 ~ 2016.10.10
# start_time='08:00:00', end_time='09:59:59'
# start_time='17:00:00', end_time='18:59:59'
# 这个数据集用于找到某一天某段时间的最相近的几个数据后, 然后在这个数据集找到对应的数据进行预测要预测的时间段
# 比如对于要预测, 2016-10-23号 8:00-10:00的流量,找到了9-22, 9-27这两天数据最相近, 然后我们需要在这个数据集找到这两天的8:00-10:00的数据
def get_offline_filter_data(freq="20Min", offline_filter_date=offline_train_or_filter_date, drop_dates=None):
    offline_filter_data = [(database.get_volume_by_time(tollgate_id, direction_id, start_time='08:00:00', end_time='09:59:59',
                                                        sumed_in_one_day=False, freq=freq, drop_dates=sw,
                                                        start_date=offline_filter_date[0], end_date=offline_filter_date[-1]),
                          database.get_volume_by_time(tollgate_id, direction_id, start_time='17:00:00', end_time='18:59:59',
                                                      sumed_in_one_day=False, freq=freq, drop_dates=xw,
                                                      start_date=offline_filter_date[0], end_date=offline_filter_date[-1]))
                           for (tollgate_id, direction_id), (sw, xw) in zip(tollgate_direction_list, drop_dates)]
    return offline_filter_data


# 线上拟合数据集, 2016.10.18 ~ 2016.10.24
# shape [5, 2, 7*2*3]   # 用的是kdd测试数据
def get_online_fit_data(freq="20Min"):
    online_fit_data = [database.get_volume_by_time_for_test(tollgate_id, direction_id, freq=freq)
                       for tollgate_id, direction_id in tollgate_direction_list]
    return online_fit_data


# 线上训练数据集,  2016.09.19 ~ 2016.10.17
def get_online_train_data(freq="20Min", online_train_date=online_train_or_filter_date, drop_dates=None):
    online_train_data = [(database.get_volume_by_time(tollgate_id, direction_id, start_time='06:00:00', end_time='07:59:59',
                                                      sumed_in_one_day=False, freq=freq, drop_dates=sw,
                                                      start_date=online_train_date[0], end_date=online_train_date[-1]),
                          database.get_volume_by_time(tollgate_id, direction_id, start_time='15:00:00', end_time='16:59:59',
                                                      sumed_in_one_day=False, freq=freq, drop_dates=xw,
                                                      start_date=online_train_date[0], end_date=online_train_date[-1]))
                         for (tollgate_id, direction_id), (sw, xw) in zip(tollgate_direction_list, drop_dates)]
    return online_train_data


# 线上训练数据集,  2016.09.19 ~ 2016.10.17
# start_time='08:00:00', end_time='09:59:59'
# start_time='17:00:00', end_time='18:59:59'
def get_online_filter_data(freq="20Min", online_filter_date=online_train_or_filter_date, drop_dates=None):
    online_filter_data = [(database.get_volume_by_time(tollgate_id, direction_id, start_time='08:00:00', end_time='09:59:59',
                                                      sumed_in_one_day=False, freq=freq, drop_dates=sw,
                                                      start_date=online_filter_date[0], end_date=online_filter_date[-1]),
                          database.get_volume_by_time(tollgate_id, direction_id, start_time='17:00:00', end_time='18:59:59',
                                                      sumed_in_one_day=False, freq=freq, drop_dates=xw,
                                                      start_date=online_filter_date[0], end_date=online_filter_date[-1]))
                          for (tollgate_id, direction_id), (sw, xw) in zip(tollgate_direction_list, drop_dates)]
    return online_filter_data


def transformed_to_standard_data(data_list):
    """
    :param data_list: predict(online=online, n_neighbors=n_neighbors)这种格式数据
    :return: 标准化预测为官方要求格式
    """
    _columns_8_10 = get_columns_2hours('08:00:00', freq='20Min')[:]
    _columns_8_10.append("10:00:00")
    _columns_17_19 = get_columns_2hours('17:00:00', freq='20Min')[:]   # deep copy
    _columns_17_19.append("19:00:00")

    columns = ['tollgate_id', 'time_window', 'direction', 'volume']
    standard_predict = pd.DataFrame(columns=columns)
    for (_tollgate_id, _direction_id), pred in zip(tollgate_direction_list, data_list):
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
    standard_predict[[columns[3]]] = standard_predict[[columns[3]]].astype(np.int64)    # 这里如果用的话,应该先round, 然后int. 但是这儿int 貌似向下取整效果更好,???
    return standard_predict


def get_test_data_standard():
    """返回官方要求的20Min interval数据格式"""
    offline_test_data = get_offline_test_data()
    test_data_list = get_test_data_list(offline_test_data)
    return transformed_to_standard_data(test_data_list)


def get_test_data_list(off_line_test_data):
    """返回的数据和predict格式一样, shape也一样"""
    res_data = []
    for filter_data_8_10, filter_data_17_19 in off_line_test_data:      # loop 5
        indexs = np.unique(filter_data_8_10.index.strftime("%Y-%m-%d"))
        filter_data_8_10 = pd.DataFrame(filter_data_8_10.values.reshape(-1, 6), index=indexs,
                                        columns=get_columns_2hours('08:00:00', freq='20Min'))
        filter_data_17_19 = pd.DataFrame(filter_data_17_19.values.reshape(-1, 6), index=indexs,
                                         columns=get_columns_2hours('17:00:00', freq='20Min'))
        res_data.append((filter_data_8_10, filter_data_17_19))
    return res_data


def write_prediction_to_csv(file_name, data):
    """
    :param file_name: 保存数据的路径
    :param data: 数据对象, 是`transformed_to_standard_predict`函数的返回
    :return:
    """
    data.to_csv(file_name, index=False)
