# -*- coding: utf-8 -*-

"""
该模型定义了一个抽象基类, 该基类用于所有的提取特征的模型
"""

from __future__ import division

import os
import sys
__project_dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__name__))))
sys.path.append(__project_dir_path)

import pandas as pd
import numpy as np

from kdd.util import PROJECT_PATH
from kdd.analysis_task_2.data import tollgate_direction_list
from kdd.analysis_task_2.data import _xw_index, _sw_index
import data as task_2_data

train_data_sets_dir = os.path.join(PROJECT_PATH, 'datasets', 'feat_dir', 'train')


class AbstractModel(object):
    """
    在父类中定义build_model() 方法
    """
    def __init__(self):
        self._model = None
        self.parameters = None

    def _predict(self):
        res_predict = []
        for toll_dire in tollgate_direction_list:
            day_predict = []
            for _sx, _sx_index in [('sw', _sw_index), ('xw', _xw_index)]:

                _file_name = str(toll_dire[0]) + "_" + str(toll_dire[1]) + "_" + _sx + ".csv"
                _file_name = os.path.join(train_data_sets_dir, _file_name)

                _data_set_df = pd.read_csv(_file_name)
                _data_set_df.set_index("Unnamed: 0", inplace=True)
                _data_set_df.index = pd.DatetimeIndex(_data_set_df.index)
                _x = _data_set_df.drop(['act', 'min_max_jg_2h_before', "max_loc_2h_before"], axis=1)
                _y = _data_set_df.act

                _train_x = _x[:-6*7]
                _train_y = _y[:-6*7]
                _test_x = _x[-6*7:]
                _test_y = _y[-6*7:]

                reg_rf = self.build_model()
                self._model = reg_rf
                reg_rf.fit(_train_x, _train_y)
                _prd = reg_rf.predict(_test_x)
                # print("&"*100)
                # a = [(i, j) for i,j in zip(reg_rf.feature_importances_, _x.columns)]
                # print(sorted(a, key=lambda x: x[0]))
                # best_param = reg_rf.best_estimator_.get_params()
                # for param_name in sorted(self.parameters.keys()):
                #     print("\t%s: %r" % (param_name, best_param[param_name]))
                # print("*"*100)

                days_list = np.unique(_test_y.index.strftime("%Y-%m-%d"))
                prd = pd.DataFrame(_prd.reshape(-1, 6), columns=_sx_index, index=days_list)
                day_predict.append(prd)
            res_predict.append(tuple(day_predict))
        return res_predict

    def predict(self):
        pred = self._predict()
        return task_2_data.transformed_to_standard_data(pred)

    def build_model(self):
        """
        :return: 返回模型
        """
        raise NotImplementedError("该函数需要在父类中实现.")
