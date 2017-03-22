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
def get_offline_fit_data(freq='20Min', offline_fit_date=offline_fit_or_test_date):
    offline_fit_data = [(database.get_volume_by_time(tollgate_id, direction_id, start_time='06:00:00', end_time='07:59:59',
                                                     sumed_in_one_day=False, freq=freq,
                                                     start_date=offline_fit_date[0], end_date=offline_fit_date[-1]),
                         database.get_volume_by_time(tollgate_id, direction_id, start_time='15:00:00', end_time='16:59:59',
                                                     sumed_in_one_day=False, freq=freq,
                                                     start_date=offline_fit_date[0], end_date=offline_fit_date[-1]))
                         for tollgate_id, direction_id in tollgate_direction_list]
    return offline_fit_data


# 线下测试数据集
# shape [5, 2, 7*2*3]  2016.10.11 ~ 2016.10.17
# start_time='08:00:00', end_time='09:59:59'
# start_time='17:00:00', end_time='18:59:59'
def get_offline_test_data(freq="20Min", offline_test_date=offline_fit_or_test_date):
    offline_test_data = [(database.get_volume_by_time(tollgate_id, direction_id, start_time='08:00:00', end_time='09:59:59',
                                                      sumed_in_one_day=False, freq=freq,
                                                      start_date=offline_test_date[0], end_date=offline_test_date[-1]),
                          database.get_volume_by_time(tollgate_id, direction_id, start_time='17:00:00', end_time='18:59:59',
                                                      sumed_in_one_day=False, freq=freq,
                                                      start_date=offline_test_date[0], end_date=offline_test_date[-1]))
                         for tollgate_id, direction_id in tollgate_direction_list]
    return offline_test_data


# 线下训练数据集,  2016.09.19 ~ 2016.10.10
def get_offline_train_data(freq="20Min", offline_train_date=offline_train_or_filter_date):
    offline_train_data = [(database.get_volume_by_time(tollgate_id, direction_id, start_time='06:00:00', end_time='07:59:59',
                                                       sumed_in_one_day=False, freq=freq,
                                                       start_date=offline_train_date[0], end_date=offline_train_date[-1]),
                          database.get_volume_by_time(tollgate_id, direction_id, start_time='15:00:00', end_time='16:59:59',
                                                      sumed_in_one_day=False, freq=freq,
                                                      start_date=offline_train_date[0], end_date=offline_train_date[-1]))
                          for tollgate_id, direction_id in tollgate_direction_list]
    return offline_train_data


# 线下过滤数据集,  2016.09.19 ~ 2016.10.10
# start_time='08:00:00', end_time='09:59:59'
# start_time='17:00:00', end_time='18:59:59'
# 这个数据集用于找到某一天某段时间的最相近的几个数据后, 然后在这个数据集找到对应的数据进行预测要预测的时间段
# 比如对于要预测, 2016-10-23号 8:00-10:00的流量,找到了9-22, 9-27这两天数据最相近, 然后我们需要在这个数据集找到这两天的8:00-10:00的数据
def get_offline_filter_data(freq="20Min", offline_filter_date=offline_train_or_filter_date):
    offline_filter_data = [(database.get_volume_by_time(tollgate_id, direction_id, start_time='08:00:00', end_time='09:59:59',
                                                        sumed_in_one_day=False, freq=freq,
                                                        start_date=offline_filter_date[0], end_date=offline_filter_date[-1]),
                          database.get_volume_by_time(tollgate_id, direction_id, start_time='17:00:00', end_time='18:59:59',
                                                      sumed_in_one_day=False, freq=freq,
                                                      start_date=offline_filter_date[0], end_date=offline_filter_date[-1]))
                          for tollgate_id, direction_id in tollgate_direction_list]
    return offline_filter_data


# 线上拟合数据集, 2016.10.18 ~ 2016.10.24
# shape [5, 2, 7*2*3]   # 用的是kdd测试数据
def get_online_fit_data(freq="20Min"):
    online_fit_data = [database.get_volume_by_time_for_test(tollgate_id, direction_id, freq=freq)
                       for tollgate_id, direction_id in tollgate_direction_list]
    return online_fit_data


# 线上训练数据集,  2016.09.19 ~ 2016.10.17
def get_online_train_data(freq="20Min", online_train_date=online_train_or_filter_date):
    online_train_data = [(database.get_volume_by_time(tollgate_id, direction_id, start_time='06:00:00', end_time='07:59:59',
                                                      sumed_in_one_day=False, freq=freq,
                                                      start_date=online_train_date[0], end_date=online_train_date[-1]),
                          database.get_volume_by_time(tollgate_id, direction_id, start_time='15:00:00', end_time='16:59:59',
                                                      sumed_in_one_day=False, freq=freq,
                                                      start_date=online_train_date[0], end_date=online_train_date[-1]))
                         for tollgate_id, direction_id in tollgate_direction_list]
    return online_train_data


# 线上训练数据集,  2016.09.19 ~ 2016.10.17
# start_time='08:00:00', end_time='09:59:59'
# start_time='17:00:00', end_time='18:59:59'
def get_online_filter_data(freq="20Min", online_filter_date=online_train_or_filter_date):
    online_filter_data = [(database.get_volume_by_time(tollgate_id, direction_id, start_time='08:00:00', end_time='09:59:59',
                                                      sumed_in_one_day=False, freq=freq,
                                                      start_date=online_filter_date[0], end_date=online_filter_date[-1]),
                          database.get_volume_by_time(tollgate_id, direction_id, start_time='17:00:00', end_time='18:59:59',
                                                      sumed_in_one_day=False, freq=freq,
                                                      start_date=online_filter_date[0], end_date=online_filter_date[-1]))
                         for tollgate_id, direction_id in tollgate_direction_list]
    return online_filter_data
