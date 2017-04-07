# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
# import pandas as pd

from .data_read import database


class graph(database):

    @staticmethod
    def show_volume_by_time(tollgate_id, direction, start_time=None, end_time=None, start_date=None, end_date=None,
                            freq='20Min', sumed_in_one_day=True, drop_dates=None, test_data=False,
                            kind_char='-', *args, **kwargs):
        """
        参数见`graph.get_volume_by_time`
        kind_char, 点的类型, 参考plt.plot 第三个参数
        """
        volumes = graph.get_volume_by_time(tollgate_id, direction, start_time=start_time, end_time=end_time,
                                           start_date=start_date, end_date=end_date, freq=freq,
                                           sumed_in_one_day=sumed_in_one_day, drop_dates=drop_dates,
                                           test_data=test_data)
        if volumes.empty:
            print("+"*50)
            print("Data from tollgate_id={0}, direction={1} is empty.".format(tollgate_id, direction))
            return

        plt.figure()
        plt.title('tollgate_id={0}, direction={1}'.format(tollgate_id, direction))
        plt.grid(True)
        if sumed_in_one_day:
            volumes.index = volumes.index.format()
            volumes.plot(grid=True)
        else:
            plt.plot(volumes.index, volumes, kind_char, *args, **kwargs)
        plt.show()

    @classmethod
    def contrast_days_by_volume(cls, args_list, draw_avg=False, tollgate_id=None, direction=None, drop_dates=None):
        """
        :param args_list: list. 每个元素的格式为(tollgate_id, direction, days)
        :param draw_avg: bool. 是否绘制出来tollgate_id, direction的日平均流量
        :param tollgate_id.  在draw_avg=True 有效.  参数详见`cls.get_volume_by_time`
        :param direction. 在draw_avg=True 有效     参数详见`cls.get_volume_by_time`
        :param drop_dates. 在draw_avg=True 有效.    参数详见`cls.get_volume_by_time`
        :return: 各图对比
        :example:
            >>>graph.contrast_days_by_volume([[2, 0, '2016-09-30'], [2, 0, '2016-09-25']],
            ...draw_avg=1, tollgate_id=2, direction=0)
        """
        from random import randint
        # from matplotlib.dates import DateFormatter
        fig = plt.figure()
        # ax1 = fig.add_subplot(111)
        # ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M:%S'))  # 设置时间标签显示格式
        # plt.xticks(rotation=90)
        len_drop_days = len(drop_dates) if drop_dates!= None else 0
        if draw_avg:
            avg_volume = cls.get_volume_by_time(tollgate_id, direction, drop_dates=drop_dates, sumed_in_one_day=True)
            # plt.xticks(pd.date_range(avg_volume.index[0], avg_volume.index[-1], freq='1min'))  # 时间间隔
            plt.plot(avg_volume.index, avg_volume/(29.0-len_drop_days),
                     label='({0}, {1})'.format(tollgate_id, direction), linewidth=3)
        for arg in args_list:
            volume = cls.get_volume_by_time(arg[0], arg[1], start_date=arg[2], end_date=arg[2], sumed_in_one_day=True)
            # plt.xticks(pd.date_range(volume.index[0], volume.index[-1], freq='1min'))  # 时间间隔
            plt.plot(volume.index, volume.values, label='({0}, {1})'.format(arg[0], arg[1])+str(arg[2]),
                     linewidth=randint(1, 2))
        plt.grid(True)
        plt.legend(loc='upper left')
        plt.show()
