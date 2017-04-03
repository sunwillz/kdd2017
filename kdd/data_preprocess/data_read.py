# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import path
from zipfile import ZipFile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ..util import PROJECT_PATH


class database(object):
    """
    和数据有关的基类。
    鉴于数据量不是很大，采用直接将数据用pandas 加载到内存
    """
    holiday = ['2016-10-01', '2016-10-02', '2016-10-03', '2016-10-04', '2016-10-05', '2016-10-06', '2016-10-07']
    abnormal_days = holiday + ['2016-09-30']
    weekend = ['2016-09-24', '2016-09-25', '2016-10-15', '2016-10-16']  # 10-8, 9 这两天因为十一是工作日

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
    def get_volume_by_time(tollgate_id, direction, start_time=None, end_time=None, start_date=None,
                           end_date=None, freq='20Min', sumed_in_one_day=True, drop_dates=None, test_data=False):
        """
        :param tollgate_id: int, 收费站id, 可选value有[1, 2, 3]
        :param direction: int, 可选value有[0, 1] 0:entry, 1:exit.    Note:tollgate 2 only have entry
        :param start_time: str, 24小时表示的时间字符串. 精确到分钟. 起始时间。如果为None, 表示从0点开始统计
            :example:  `18:36`    表示18点36分钟
        :param end_time:  同参数start_time, 结束时间. 如果为None,表示截止到24为止
        :param start_date:  str, 开始统计日期。 如果为None则表示从最早数据所在时间开始
        :param end_date:  同参数start_date。 如果为None则表示以最晚数据截止
        :param freq:  该参数同page http://pandas.pydata.org/pandas-docs/stable/timeseries.html  上的用法
        :param sumed_in_one_day: bool.  True表示统计所有日期这个时间段的数据
        :param drop_dates: list. 需要drop掉的数据
        :param test_data: bool.  是否用测试数据集
        :return: pandas.Series() 按照freq, sumed_in_one_day参数group 之后的流量。 if sumed_in_one_day=False, 则返回

        :Notes: 如果test_data=True, 请调用, database.get_volume_by_time_for_test
        :Notes: 如果想返回一天的数据，那么只需将start_date和end_date设置为这一天就行
            :example: 获取2016-10-06这天tollgate_id=1, direction=0的数据
                      `get_volume_by_time(1, 0, start_date='2016-10-06', end_date='2016-10-06')`
        """
        volumes = database.train_volume if not test_data else database.test_volume
        volumes = volumes[(volumes['tollgate_id']==tollgate_id) & (volumes['direction']==direction)]
        # 上面过滤和下面设置time列为索引不可以颠倒顺序，否则出现reindex 类型错误
        volumes = volumes.set_index('time').sort_index()    # 将流量表转化为按照time列为索引的DataFrame

        start_date = volumes.index[0].strftime('%Y-%m-%d') if not start_date else start_date
        end_date = volumes.index[-1].strftime('%Y-%m-%d') if not end_date else end_date

        # 因为tollgate_id 不缺失，故可以用这一列作为流量返回值  # series
        volumes_count = volumes.resample(freq).count()['tollgate_id']      # 按照freq频率统计volume

        start_time = '00:00:00' if not start_time else start_time
        end_time = '23:59:59' if not end_time else end_time
        volumes_count = volumes_count.loc[start_date:end_date]    # 截取需要时间的数据
        # 有一个bug, 需要分别截取每天的位于两个time段的数据
        dt_series = pd.period_range(start_date, end_date, freq='D')  # 返回每天的日期 ‘2016-09-28’
        if sumed_in_one_day and not volumes_count.empty:
            time_stamp = [str(t).split(' ')[-1] for t in
                          volumes_count[start_date].loc[
                          start_date + " " + start_time : start_date+" "+end_time].index
                          ]   # 返回一天以内start_time,到end_time的每隔freq的时间戳
            # res = pd.Series([0]*len(time_stamp), index=time_stamp)
            res = np.zeros(len(time_stamp))
            for dt in dt_series:
                every_day_data = volumes_count[str(dt)+" "+start_time : str(dt)+" "+end_time]   # 得到dt 这天start_time到end_time的数据
                if every_day_data.shape == res.shape:
                    res += every_day_data.values
                else:
                    raise ValueError('some thing wrong, when day={day}, there are {day_count} '
                                     'data while normal {norm_count} data'.format(day=dt, day_count=every_day_data.shape[0],
                                                                                  norm_count=res.shape[0]))
            # volumes_count = pd.Series(res, index=time_stamp)    # 返回所有天的以freq统计的流量和的数据
            volumes_count = pd.Series(res, index=pd.TimedeltaIndex(time_stamp))    # 返回所有天的以freq统计的流量和的数据
        elif not sumed_in_one_day and not volumes_count.empty:
            all_index = []
            for dt in dt_series:
                all_index.extend(pd.period_range(str(dt)+" "+start_time, str(dt)+" "+end_time, freq=freq))
            all_index = [str(i) for i in all_index]
            volumes_count = volumes_count[volumes_count.index.isin(all_index)]
        else:
            return pd.Series()

        # 去掉drop_dates里面的数据
        drop_dates = [] if not drop_dates else drop_dates
        for drop_date in drop_dates:
            tomorrow = datetime.strptime(drop_date, "%Y-%m-%d") + timedelta(hours=24)
            drop_indexs = pd.period_range(drop_date + ' 00:00:00', tomorrow, freq=freq)
            drop_indexs.drop([drop_indexs[0], drop_indexs[-1]])
            drop_indexs = [pd.Timestamp(str(d) + ":00") for d in drop_indexs]  # 每个元素为类似 `2016-01-01 00:20:00`
            volumes_count = volumes_count.drop(drop_indexs, errors='ignore')
        assert volumes_count.isnull().sum() == 0, "there are nan in data. tollgate_id={0}, direction={1}".format(tollgate_id, direction)     # 断言数据中是否存在nan
        return volumes_count

    @staticmethod
    def get_volume_by_time_for_test(tollgate_id, direction, freq="20Min"):
        volume_6_8 = database.get_volume_by_time(tollgate_id, direction, start_time='06:00:00', end_time='07:59:59',
                                                 sumed_in_one_day=False, test_data=True, freq=freq)    # 设置成8:00 会统计8:00-8:20的
        volume_15_17 = database.get_volume_by_time(tollgate_id, direction, start_time='15:00:00', end_time='16:59:59',
                                                   sumed_in_one_day=False, test_data=True, freq=freq)
        return volume_6_8, volume_15_17

    @staticmethod
    def volume_to_csv(tollgate_id, direction, start_time=None, end_time=None, start_date=None,
                      end_date=None, freq='20Min', sumed_in_one_day=False, drop_dates=None,
                      data_dir_path=PROJECT_PATH, file_name=None):
        """
        data_dir_path: 保存数据的路径
        file_name: 保存数据的文件名
        参数同database.get_volume_by_time
        """
        volume = database.get_volume_by_time(tollgate_id, direction, start_time=start_time, end_time=end_time,
                                             start_date=start_date, end_date=end_date, freq=freq,
                                             sumed_in_one_day=sumed_in_one_day, drop_dates=drop_dates)
        # save data_to_csv
        if not file_name:
            file_name = "volume_" + "tollgate" + str(tollgate_id) + "_direction" + str(direction)+"_"+freq
            if start_date or end_date or not len(drop_dates):   # 过滤了日期了
                file_name += "_drop_date"
            if start_time or end_time:
                file_name += "_drop_time"
            file_name += '.csv'
        abs_path_file_name = path.join(data_dir_path, file_name)
        volume.to_csv(abs_path_file_name)

    @staticmethod
    def volume_all_to_csv(start_time=None, end_time=None, start_date=None, end_date=None, freq='20Min',
                          sumed_in_one_day=False, drop_dates=None, data_dir_path=PROJECT_PATH):
        """
        参数同 database.volume_to_csv
        """
        # 0:entry, 1:exit
        for tollgate_id, direction_id in [(1, 0), (1, 1), (2, 0), (3, 0), (3, 1)]:  # tollgate 2 only have entry
            database.volume_to_csv(tollgate_id, direction_id, start_time=start_time, end_time=end_time,
                                   start_date=start_date, end_date=end_date, freq=freq,
                                   sumed_in_one_day=sumed_in_one_day, drop_dates=drop_dates,
                                   data_dir_path=data_dir_path)
