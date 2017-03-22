# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# 项目的所在路径, 即包含datasets文件夹的那个kdd文件夹, 这个kdd 文件夹为最外面的那个文件夹
PROJECT_PATH = os.path.dirname(os.path.abspath(__name__))


def parse_freq(freq):
    """解析freq参数, 这儿是固定的2Hours"""
    _temp = filter(str.isdigit, freq)
    _temp = 1 if _temp == '' else int(_temp)
    if 120 % _temp != 0:
        raise ValueError("freq=" + freq + "不可以被120整除, 不可以作为间歇分隔")
    if freq.endswith('s'):
        # interval 作为分割间歇， 比如freq='20Min', 那么interval为6。 如果freq='30s', 那么interval为240
        interval = (60 / _temp) * 120
    elif freq.endswith('Min'):
        interval = 120 / _temp
    return int(interval)
