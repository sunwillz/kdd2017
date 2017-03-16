# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import path
from zipfile import ZipFile
from datetime import datetime, timedelta
from dateutil import relativedelta

import numpy as np
import pandas as pd

from ..util import PROJECT_PATH


class database(object):
    """
    和数据有关的基类。
    鉴于数据量不是很大，采用直接将数据用pandas 加载到内存
    """
    _data_dir = path.join(PROJECT_PATH, 'datasets')     # 包含dataSets.zip 的那个文件夹
    _zip_file = path.join(_data_dir, 'dataSets.zip')
    with ZipFile(_zip_file) as my_zip:

        # 将数据以pandas.DataFrame() 方式存储
        with my_zip.open('dataSets/training/links (table 3).csv') as f:
            # [u'link_id', u'length', u'width', u'lanes', u'in_top', u'out_top', u'lane_width']
            train_links = pd.read_csv(f)

        with my_zip.open('dataSets/training/routes (table 4).csv') as f:
            # [u'intersection_id', u'tollgate_id', u'link_seq']
            train_routes = pd.read_csv(f)

        with my_zip.open('dataSets/training/trajectories(table 5)_training.csv') as f:
            # 将starting_time 这一列类型转化为timestamp类型便于时间上的操作
            # [u'intersection_id', u'tollgate_id', u'vehicle_id', u'starting_time', u'travel_seq', u'travel_time']
            train_trajectories = pd.read_csv(f, parse_dates=['starting_time'])

        with my_zip.open('dataSets/training/volume(table 6)_training.csv') as f:
            # [u'time', u'tollgate_id', u'direction', u'vehicle_model', u'has_etc', u'vehicle_type']
            train_volume = pd.read_csv(f, parse_dates=['time'])

        with my_zip.open('dataSets/training/weather (table 7)_training.csv') as f:
            # columns = [u'pressure', u'sea_pressure', u'wind_direction',
            #  u'wind_speed', u'temperature', u'rel_humidity', u'precipitation']
            # index = ['date', 'hour']
            train_weather = pd.read_csv(f, parse_dates=['date'], index_col=['date', 'hour'])

        with my_zip.open('dataSets/testing_phase1/trajectories(table 5)_test1.csv') as f:
            # [u'intersection_id', u'tollgate_id', u'vehicle_id', u'starting_time', u'travel_seq', u'travel_time']
            test_trajectories = pd.read_csv(f, parse_dates=['starting_time'])

        with my_zip.open('dataSets/testing_phase1/volume(table 6)_test1.csv') as f:
            # [u'time', u'tollgate_id', u'direction', u'vehicle_model', u'has_etc', u'vehicle_type']
            test_volume = pd.read_csv(f, parse_dates=['time'])

        with my_zip.open('dataSets/testing_phase1/weather (table 7)_test1.csv') as f:
            # columns = [u'pressure', u'sea_pressure', u'wind_direction',
            #  u'wind_speed', u'temperature', u'rel_humidity', u'precipitation']
            # index = ['date', 'hour']
            test_weather = pd.read_csv(f, parse_dates=['date'], index_col=['date', 'hour'])

    @staticmethod
    def get_vehicle_travel_time(intersection_id, tollgate_id, starting_time, precipitation):
        """
        :param intersection_id: str, 十字路口id, 可选value有[A, B, C]
        :param tollgate_id: int, 收费站id, 可选value有[1, 2, 3]
        :param starting_time: str or timestamp,
        :param precipitation: tuple, len(precipitation)=2,降雨量过滤
                precipitation[0]表示降雨量下界, precipitation[1] 表示降雨量上界

        :return: 在intersection_id和tollgate_id之间的travel_time         # 还未定义返回的数据结构
        """
        pass

    @staticmethod
    def get_link_average_speed(direction=-1, **kwargs):
        """
        :param direction: int, 0:entry, 1:exit. -1:all
        :param kwargs: dict

        :return: pandas.DataFrame(). columns=['link_id', 'direction', 'avg_speed']  # 还未定义好
        """
        pass

    @staticmethod
    def get_volume_by_time(tollgate_id, direction, start_time=None, end_time=None, start_date=None,
                           end_date=None, freq='20Min', sumed_in_one_day=True, drop_dates=None):
        """
        :param tollgate_id: int, 收费站id, 可选value有[1, 2, 3]
        :param direction: int, 可选value有[0, 1] 0:entry, 1:exit.
        :param start_time: str, 24小时表示的时间字符串. 精确到分钟. 起始时间。如果为None, 表示从0点开始统计
            :example:  `18:36`    表示18点36分钟
        :param end_time:  同参数start_time, 结束时间. 如果为None,表示截止到24为止
        :param start_date:  str, 开始统计日期。 如果为None则表示从最早数据所在时间开始
        :param end_date:  同参数start_date。 如果为None则表示以最晚数据截止
        :param freq:  该参数同page http://pandas.pydata.org/pandas-docs/stable/timeseries.html  上的用法
        :param sumed_in_one_day: bool.  True表示统计所有日期这个时间段的数据
        :param drop_dates: list. 需要drop掉的数据
        :return: pandas.Series() 按照freq, sumed_in_one_day参数group 之后的流量

        :Notes: 如果想返回一天的数据，那么只需将start_date和end_date设置为这一天就行
            :example: 获取2016-10-06这天tollgate_id=1, direction=0的数据
                      `get_volume_by_time(1, 0, start_date='2016-10-06', end_date='2016-10-06')`
        """
        train_volumes = database.train_volume
        train_volumes = train_volumes[(train_volumes['tollgate_id']==tollgate_id)
                                      & (train_volumes['direction']==direction)]
        # 上面过滤和下面设置time列为索引不可以颠倒顺序，否则出现reindex 类型错误
        train_volumes = train_volumes.set_index('time').sort_index()    # 将流量表转化为按照time列为索引的DataFrame

        start_date = train_volumes.index[0].strftime('%Y-%m-%d') if not start_date else start_date
        end_date = train_volumes.index[-1].strftime('%Y-%m-%d') if not end_date else end_date

        # 因为tollgate_id 不缺失，故可以用这一列作为流量返回值  # series
        train_volumes_count = train_volumes.resample(freq).count()['tollgate_id']      # 按照freq频率统计volume

        start_time = '00:00:00' if not start_time else start_time
        end_time = '23:59:59' if not end_time else end_time
        start = start_date + " " + start_time
        end = end_date + " " + end_time
        train_volumes_count = train_volumes_count.loc[start:end]    # 截取需要时间的数据

        drop_dates = [] if not drop_dates else drop_dates
        for drop_date in drop_dates:
            tomorrow = datetime.strptime(drop_date, "%Y-%m-%d") + timedelta(hours=24)
            drop_indexs = pd.period_range(drop_date+' 00:00:00', tomorrow, freq=freq)
            drop_indexs.drop([drop_indexs[0], drop_indexs[-1]])
            drop_indexs = [pd.Timestamp(str(d)+":00") for d in drop_indexs]   # 每个元素为类似 `2016-01-01 00:20:00`
            train_volumes_count = train_volumes_count.drop(drop_indexs, errors='ignore')
        if sumed_in_one_day and not train_volumes_count.empty:
            time_stamp = [str(t).split(' ')[-1] for t in train_volumes_count[start_date].index]   # 返回一天以内的每隔freq的时间戳
            dt_series = pd.period_range(start_date, end_date, freq='D')    # 返回每天的日期 ‘2016-09-28’
            # res = pd.Series([0]*len(time_stamp), index=time_stamp)
            res = np.zeros(len(time_stamp))
            for dt in dt_series:
                every_day_data = train_volumes_count[str(dt)]
                if every_day_data.shape == res.shape:
                    res += every_day_data.values
                else:
                    raise ValueError('有数据缺失')
            train_volumes_count = pd.Series(res/len(dt_series), index=time_stamp)    # 返回平均每一天的freq的数据
        return train_volumes_count

