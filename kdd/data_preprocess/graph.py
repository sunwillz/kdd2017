# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt

from .data_read import database


class graph(database):

    @staticmethod
    def show_volume_by_time(tollgate_id, direction, start_time=None, end_time=None, start_date=None, end_date=None,
                            freq='20Min', sumed_in_one_day=True, drop_dates=None, kind_char='-', *args, **kwargs):
        """
        参数见`graph.get_volume_by_time`
        kind_char, 点的类型, 参考plt.plot 第三个参数
        """
        volumes = graph.get_volume_by_time(tollgate_id, direction, start_time=start_time, end_time=end_time,
                                           start_date=start_date, end_date=end_date, freq=freq,
                                           sumed_in_one_day=sumed_in_one_day, drop_dates=drop_dates)
        if volumes.empty:
            print("+"*50)
            print("Data from tollgate_id={0}, direction={1} is empty.".format(tollgate_id, direction))
            return

        plt.figure()
        plt.title('tollgate_id={0}, direction={1}'.format(tollgate_id, direction))
        plt.grid(True)
        plt.plot(volumes.index, volumes, kind_char, *args, **kwargs)
        plt.show()

