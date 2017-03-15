# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import path
from zipfile import ZipFile

import pandas as pd

from ..util import PROJECT_PATH


class DataBase(object):
    """
    和数据有关的基类。
    鉴于数据量不是很大，采用直接将数据用pandas 加载到内存
    """
    _data_dir = path.join(PROJECT_PATH, 'datasets')     # 包含dataSets.zip 的那个文件夹
    _zip_file = path.join(_data_dir, 'dataSets.zip')
    with ZipFile(_zip_file) as my_zip:

        # 将数据以pandas.DataFrame() 方式存储
        with my_zip.open('dataSets/training/links (table 3).csv') as f:
            train_links = pd.read_csv(f)

        with my_zip.open('dataSets/training/routes (table 4).csv') as f:
            train_routes = pd.read_csv(f)

        with my_zip.open('dataSets/training/trajectories(table 5)_training.csv') as f:
            # 将starting_time 这一列类型转化为timestamp类型便于时间上的操作
            train_trajectories = pd.read_csv(f, parse_dates=['starting_time'])

        with my_zip.open('dataSets/training/volume(table 6)_training.csv') as f:
            train_volume = pd.read_csv(f, parse_dates=['time'])

        with my_zip.open('dataSets/training/weather (table 7)_training.csv') as f:
            train_weather = pd.read_csv(f, parse_dates=['date'])

        with my_zip.open('dataSets/testing_phase1/trajectories(table 5)_test1.csv') as f:
            test_trajectories = pd.read_csv(f, parse_dates=['starting_time'])

        with my_zip.open('dataSets/testing_phase1/volume(table 6)_test1.csv') as f:
            test_volume = pd.read_csv(f, parse_dates=['time'])

        with my_zip.open('dataSets/testing_phase1/weather (table 7)_test1.csv') as f:
            test_weather = pd.read_csv(f, parse_dates=['date'])

