# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:34:48 2017

@author: sunwill
"""

import numpy as np
import pandas as pd

##task1:ETA评测函数
def task1_eva_metrics(predict, actual):
#    predcit是预测结果文件，actual是真实结果文件，其数据结构如下：
#    intersection_id:string,inetersection ID
#    tollgate_id:string,tollgate ID
#    time_window:string,[2016-09-18 23:40:00, 2016-09-19 00:00:00)
#    avg_travel_time:float,average travel time(second)

    predcit_data = pd.read_csv(predict,names=['intersection_id','tollgate_id','time_window','ETA_pre'])
    actual_data = pd.read_csv(actual,names=['intersection_id','tollgate_id','time_window','ETA_actual'])
    
    df = pd.merge(predcit_data,actual_data,on=['intersection_id','tollgate_id','time_window'])
    df['ETA_pre'].astype(float)
    df['ETA_actual'].astype(float)
    df['diff'] = (np.abs(df['ETA_pre']-df['ETA_actual']))/df['ETA_actual']
    grouped_by_id = df.groupby(['intersection_id','tollgate_id'])['diff'].mean()
    grouped_by_id = grouped_by_id.reset_index()
    MAPE = grouped_by_id['diff'].mean()
   
    return MAPE

    

##task2:流量预测 评测函数
def task2_eva_metrics(predict, actual):
#    predicts是预测结果文件，actual是真实结果文件，其数据结构如下：
#    tollgate_id:string, tollgate ID
#    time_window:string,[2016-09-18 23:40:00, 2016-09-19 00:00:00)
#    direction:string,0:entry,1:exit
#    volume:int,total volume
    predict_data = pd.read_csv(predict,names=['tollgate_id','time_window','direction','pre_volume'])
    actual_data = pd.read_csv(actual,names=['tollgate_id','time_window','direction','act_volume'])
    df = pd.merge(predict_data,actual_data,on=['tollgate_id','time_window','direction'])
    df['pre_volume'].astype(int)
    df['act_volume'].astype(int)
    df['diff'] = (np.abs(df['pre_volume']-df['act_volume']))/df['act_volume']
    grouped_by_id = df.group_by(['tollgate_id','direction'])['diff'].mean()
    grouped_by_id = grouped_by_id.reset_index()
    MAPE = grouped_by_id['diff'].mean()
    
    return MAPE