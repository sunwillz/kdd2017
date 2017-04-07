#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
运行该文件获取特征, 将保存在kdd/datasets/task2文件夹下面.
    volume(table 6)_training.csv
    weather(table 7)_training_update.csv

    volume表:
        横向提取  (即提取同一天6-8的特征)
            20分钟窗口 (6个特征) [使上午和下午表现一致, 窗口统一用wind[1-6]_2h_before表示.      wind[1-6]_2h_before
            20分钟中位数                                                                   median_2h_before
            20分钟数据标准差                                                               std_2h_before
            20分钟最大值                                                                   max_2h_before
            20分钟最小值                                                                   min_2h_before
            20分钟最大值出现在第几个窗口上(可选值 1,2,3,4,5,6)                                max_loc_2h_before
            20分钟最大值与最小值所在坐标的间隔                                                min_max_jg_2h_before

        纵向提取  (比如提取历史上的8:00-8:20特征)
            同窗口历史中位数                                                                wind_hist_median
            同窗口历史均值                                                                  wind_hist_mean
            同窗口同周x中位数                                                               same_weekx_median
            同窗口最近5天中位数                                                             wind_5d_mean
            同窗口上周同时间前三天的均值                                                      wind_last3_weekx_mean
            同窗口历史has_etc所占比例同历史所有的均值比例的比例 (如该窗口为0.24, 历史均值为0.21, 比例为0.24/0.21)
                                                                                          etc_rate_div_avg
            同窗口所在周x历史has_etc所占比例同历史所有的均值比例的比例 (如该窗口为0.24, 历史均值为0.21, 比例为0.24/0.21)
                                                                                          weekx_etc_rate_div_avg

    weather表:
            要预测窗口内降雨情况(没雨, 小雨, 大雨, 暴雨) ->(0,1,2,3)                          precipitation
"""

from __future__ import division

from datetime import datetime, timedelta
import os
import sys
__project_dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__name__))))
sys.path.append(__project_dir_path)

import pandas as pd
import numpy as np

from kdd.data_preprocess import database
from kdd.analysis_task_2.data import tollgate_direction_list
from kdd.analysis_task_2.data import online_train_or_filter_date as train_dates
from kdd.analysis_task_2.data import online_fit_date as test_dates
from kdd.analysis_task_2.data import _sw_index, _xw_index


sw = ("06:00:00", "07:59:59")
sw_prd_wind = ("08:00:00", "09:59:59")
xw = ("15:00:00", "16:59:59")
xw_prd_wind = ("17:00:00", "18:59:59")
# ++++++++++++++++++++++++++++++ volume表, 横向提取 ++++++++++++++++++++++++++++++++++
def extract_2h_before(is_train_data):
    """以20分钟窗口抽取特征
    :param is_train_data: boolean.   是不是抽取训练数据集里面的钱两小时特征
    """
    df_mat = []     #
    for _tollgate_id, _direction in tollgate_direction_list:
        _day = []
        for _sx in [sw, xw]:
            if is_train_data:
                volume = database.get_volume_by_time(_tollgate_id, _direction, start_time=_sx[0], end_time=_sx[1],
                                                     sumed_in_one_day=False, test_data=False)
            else:
                volume = database.get_volume_by_time(_tollgate_id, _direction, start_time=_sx[0], end_time=_sx[1],
                                                     sumed_in_one_day=False, test_data=True)
            days_list = np.unique(volume.index.strftime("%Y-%m-%d"))
            volume = volume.values.reshape(-1, 6)   # flat
            volume_df = pd.DataFrame(volume, index=days_list, columns=["wind"+str(i)+"_2h_before" for i in range(1, 7)])  # wind[1-6]
            volume_df["max_2h_before"] = volume_df.max(axis=1)
            volume_df["min_2h_before"] = volume_df.min(axis=1)
            volume_df["median_2h_before"] = volume_df.median(axis=1)
            volume_df["std_2h_before"] = volume_df.std(axis=1)
            vol_max_loc, vol_min_loc = [], []
            for i in range(volume_df.shape[0]):
                vol_max_loc.append(int(volume_df.iloc[i].argmax()[4]))  # 获取windi_2h_before 第四个字符
                vol_min_loc.append(int(volume_df.iloc[i].argmin()[4]))
            volume_df["max_loc_2h_before"] = vol_max_loc
            volume_df["min_max_jg_2h_before"] = [_max-_min for _max, _min in zip(vol_max_loc, vol_min_loc)]
            _day.append(volume_df)
        df_mat.append(tuple(_day))
    return df_mat

# ++++++++++++++++++++++++++++++volume表, 纵向提取 +++++++++++++++++++++++++++++++++++
def extract_hist_feature_for_test():
    """
    :return: 返回一个列表, shape=(5, 2, 7) 每个元素为一天的DataFrame, index=[wind1-6], columns为提取的特征
    """
    wind_extract_list = []
    for toll_di in tollgate_direction_list:
        _day = []
        for _sx, _sx_index in [(sw_prd_wind, _sw_index), (xw_prd_wind, _xw_index)]:
            _days_feature_df = []
            for _date in pd.date_range(test_dates[0], test_dates[1]):
                volume = database.get_volume_by_time(toll_di[0], toll_di[1], start_time=_sx[0], end_time=_sx[1],
                                                     sumed_in_one_day=False)
                days_list = np.unique(volume.index.strftime("%Y-%m-%d"))
                volume = volume.values.reshape(-1, 6)  # flat
                volume = pd.DataFrame(volume, index=days_list, columns=_sx_index)
                wind_df = pd.DataFrame(index=_sx_index)
                wind_df["wind_hist_median"] = volume.median(axis=0)
                wind_df["wind_hist_mean"] = volume.mean(axis=0)
                normal_days = pd.date_range(train_dates[0], train_dates[1]).strftime("%Y-%m-%d")
                history_week_x = [i for i in normal_days if _date.weekday() == datetime.strptime(i, "%Y-%m-%d").weekday()]
                wind_df["same_weekx_median"] = volume.loc[history_week_x, :].median(axis=0)
                wind_df["wind_5d_mean"] = volume.tail(5).median(axis=0)
                last_week_3day = [(_date + timedelta(days=-i)).strftime("%Y-%m-%d") for i in (7, 8, 9)]
                wind_df["wind_last3_weekx_mean"] = volume.loc[last_week_3day, :].median(axis=0)
                etc_hist_rate = database.train_volume.has_etc.sum()/database.train_volume.has_etc.count()
                #  这儿还有  etc_rate_div_avg, weekx_etc_rate_div_avg 两个特征没提取
                _days_feature_df.append(wind_df)
            _day.append(tuple(_days_feature_df))
        wind_extract_list.append(tuple(_day))
    return wind_extract_list


def extract_hist_feature_for_train():
    """
    和for_test的区别在于这个地方需要填补缺失值, 即对于历史周x, 最开始的几天没有这个数据
    :return: 返回一个列表, shape=(5, 2, 7) 每个元素为一天的DataFrame, index=[wind1-6], columns为提取的特征。
    """
    wind_extract_list = []
    for toll_di in tollgate_direction_list:
        _day = []
        for _sx, _sx_index in [(sw_prd_wind, _sw_index), (xw_prd_wind, _xw_index)]:
            _days_feature_df = []
            for _i, _date in enumerate(pd.date_range(train_dates[0], train_dates[1])):
                volume = database.get_volume_by_time(toll_di[0], toll_di[1], start_time=_sx[0], end_time=_sx[1],
                                                     sumed_in_one_day=False)
                yesterday = (_date + timedelta(days=-1)).strftime("%Y-%m-%d")
                days_list = np.unique(volume.index.strftime("%Y-%m-%d"))
                volume = volume.values.reshape(-1, 6)  # flat
                volume = pd.DataFrame(volume, index=days_list, columns=_sx_index)
                all_median = volume.median(axis=0)     # 这个用来补全中位数缺失值
                all_mean = volume.mean(axis=0)
                volume = volume.loc[:yesterday, :]      # 这个地方是截取该天之前的所有的历史数据
                wind_df = pd.DataFrame(index=_sx_index)
                wind_df["wind_hist_median"] = volume.median(axis=0) if not volume.empty else all_median
                wind_df["wind_hist_mean"] = volume.mean(axis=0) if not volume.empty else all_mean
                normal_days = pd.date_range(train_dates[0], yesterday).strftime("%Y-%m-%d")
                history_week_x = [i for i in normal_days if _date.weekday() == datetime.strptime(i, "%Y-%m-%d").weekday()]
                wind_df["same_weekx_median"] = volume.loc[history_week_x, :].median(axis=0) \
                    if not volume.loc[history_week_x, :].empty else all_median
                wind_df["wind_5d_mean"] = volume.tail(5).median(axis=0) if not volume.tail(5).empty else all_median
                last_week_3day = [(_date + timedelta(days=-i)).strftime("%Y-%m-%d") for i in (7, 8, 9)
                                  if (_date + timedelta(days=-i)).strftime("%Y-%m-%d") in normal_days]
                wind_df["wind_last3_weekx_mean"] = volume.loc[last_week_3day, :].median(axis=0) \
                    if not volume.loc[last_week_3day, :].empty else all_median
                etc_hist_rate = database.train_volume.has_etc.sum()/database.train_volume.has_etc.count()
                #  这儿还有  etc_rate_div_avg, weekx_etc_rate_div_avg 两个特征没提取
                _days_feature_df.append(wind_df)
            _day.append(tuple(_days_feature_df))
        wind_extract_list.append(tuple(_day))
    return wind_extract_list

# ++++++++++++++++++++++++++++++weather表 +++++++++++++++++++++++++++++++++++++++++++
def extract_wind_precipitation(is_train_data):
    if is_train_data:
        precipitation = database.train_weather.precipitation
        dates_range = pd.date_range(train_dates[0], train_dates[1])
    else:
        precipitation = database.test_weather.precipitation
        dates_range = pd.date_range(test_dates[0], test_dates[1])


def construct_feature_matrix(df_mat_list, hist_wind_feature_list, is_train_data=False):
    # feat_mat_list = []
    for _road_df_mat_list, _hist_wind_feature_list, toll_dirc in zip(df_mat_list, hist_wind_feature_list, tollgate_direction_list):
        # _day_mat_list = []
        for _sx_2hour_before_mat, _sx_hist_wind_list, _sx, _time in zip(_road_df_mat_list, _hist_wind_feature_list, [sw_prd_wind, xw_prd_wind], ['sw', 'xw']):
            _volume_label = database.get_volume_by_time(toll_dirc[0], toll_dirc[1], start_time=_sx[0], end_time=_sx[1],
                                                        sumed_in_one_day=False)
            days_list = np.unique(_volume_label.index.strftime("%Y-%m-%d"))
            feature_df = pd.DataFrame(columns=list(_sx_2hour_before_mat.columns.values) + list(_sx_hist_wind_list[0].columns.values))
            for _i, _wind in enumerate(_sx_hist_wind_list):
                _a_feat_2h = _sx_2hour_before_mat.iloc[_i]
                for _index in _wind.index:
                    _aa = list(_a_feat_2h.values) + list(_wind.loc[_index].values)
                    _feat_index = days_list[_i] + " " + _index
                    feature_df.loc[_feat_index] = _aa    # 将数据追加到DataFrame末尾
            if is_train_data:
                # _volume_data = _volume_label.values.reshape(-1, 6)  # flat
                # volume_label = pd.DataFrame(_volume_data, index=days_list, columns=['wind'+str(i) for i in range(1, 7)])
                feature_df["act"] = _volume_label.values
                _feat_dir = os.path.join(__project_dir_path, "datasets", "feat_dir", "train")
            else:
                _feat_dir = os.path.join(__project_dir_path, "datasets", "feat_dir", "test")
            # save feature matrix to csv
            if not os.path.exists(_feat_dir):
                os.makedirs(_feat_dir)
            _file_name = str(toll_dirc[0]) + "_" + str(toll_dirc[1]) + "_" + _time + '.csv'
            feature_df.to_csv(os.path.join(_feat_dir, _file_name))

            # _day_mat_list.append(feature_df)
        # feat_mat_list.append(tuple(_day_mat_list))
    # return feat_mat_list


if __name__ == "__main__":
    # train volume data_set
    train_df_mat_list = extract_2h_before(is_train_data=True)
    train_hist_wind_feature_list = extract_hist_feature_for_train()
    construct_feature_matrix(train_df_mat_list, train_hist_wind_feature_list, is_train_data=True)

    # test volume data_set
    test_df_mat_list = extract_2h_before(is_train_data=False)
    test_hist_wind_feature_list = extract_hist_feature_for_test()
    construct_feature_matrix(test_df_mat_list, test_hist_wind_feature_list, is_train_data=False)
