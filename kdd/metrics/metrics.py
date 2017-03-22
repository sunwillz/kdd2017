# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:34:48 2017

@author: sunwill
"""

import numpy as np
import pandas as pd

##task1:ETA评测函数
def task1_eva_metrics(predict, actual):
    """
#    predict是预测结果文件，actual是真实结果文件. 或者predict, actual为pandas.DataFrame对象.其数据结构如下：
#    intersection_id:string,inetersection ID
#    tollgate_id:string,tollgate ID
#    time_window:string,[2016-09-18 23:40:00, 2016-09-19 00:00:00)
#    avg_travel_time:float,average travel time(second)
    """
    if isinstance(predict, basestring) or isinstance(actual, basestring):
        predict_data = pd.read_csv(predict)
        actual_data = pd.read_csv(actual)
    else:
        predict_data = predict
        actual_data = actual
    predict_data.rename(index=str, columns={"ETA": "ETA_pre"}, inplace=True)
    actual_data.rename(index=str, columns={"ETA": "ETA_actual"}, inplace=True)
    df = pd.merge(predict_data,actual_data,on=['intersection_id','tollgate_id','time_window'])
    df['ETA_pre'].astype(float)
    df['ETA_actual'].astype(float)
    df['diff'] = (np.abs(df['ETA_pre']-df['ETA_actual']))/df['ETA_actual']
    grouped_by_id = df.groupby(['intersection_id','tollgate_id'])['diff'].mean()
    grouped_by_id = grouped_by_id.reset_index()
    MAPE = grouped_by_id['diff'].mean()
   
    return MAPE


##task2:流量预测 评测函数
def task2_eva_metrics(predict, actual):
    """
#    predicts是预测结果文件，actual是真实结果文件. 或者predict, actual为pandas.DataFrame对象. 其数据结构如下：
#    tollgate_id:string, tollgate ID
#    time_window:string,[2016-09-18 23:40:00, 2016-09-19 00:00:00)
#    direction:string,0:entry,1:exit
#    volume:int,total volume
    """
    if isinstance(predict, basestring) or isinstance(actual, basestring):
        predict_data = pd.read_csv(predict)
        actual_data = pd.read_csv(actual)
    else:
        predict_data = predict
        actual_data = actual
    predict_data.rename(index=str, columns={"volume": "pre_volume"}, inplace=True)
    actual_data.rename(index=str, columns={"volume": "act_volume"}, inplace=True)
    df = pd.merge(predict_data,actual_data,on=['tollgate_id','time_window','direction'])
    df['pre_volume'].astype(int)
    df['act_volume'].astype(int)
    df['diff'] = (np.abs(df['pre_volume']-df['act_volume']))/df['act_volume']
    grouped_by_id = df.groupby(['tollgate_id','direction'])['diff'].mean()
    grouped_by_id = grouped_by_id.reset_index()
    MAPE = grouped_by_id['diff'].mean()
    
    return MAPE
