#!/user/bin/env python
# -*- coding: UTF-8 -*-
"""
-------------------------------
@Author : zspp
-------------------------------
@File : work_load_DDL9-- process_data.py
@Software : PyCharm
@Time : 2022/6/11 0:25
-------------------------------
"""
import os
import pandas as pd
import numpy as np
import random


def process_300(all_data_nums):
    """
    处理数据，从大数据集中选取统一的训练+测试大小数量
    :return:
    """
    for f in os.listdir('data'):
        # print(f)
        systerm_name = ''.join(f.split('_')[1:]).lower().split('.')[0]
        print(systerm_name)
        if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           'prepare_data')):
            os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'prepare_data'))

        # 截取300条数据
        data = pd.read_csv(os.path.join(os.getcwd(), 'data/{}'.format(f)))

        columns = list(data.columns)
        columns[-1] = 'PERF'
        data.columns = columns

        # # 删除异常值
        data = data.drop(data[data['PERF'] == -1].index)
        print(data)
        data = data.reset_index(drop=True)

        clist = random.sample(range(data.shape[0]), all_data_nums)
        data = data.iloc[clist, :]
        data = data.reset_index(drop=True)

        data.to_csv('./prepare_data/{}_random_{}.csv'.format(systerm_name, all_data_nums), index=False)


def make_test_data(all_data_nums,test_size):
    """
    制造测试数据集
    """
    for f in os.listdir('prepare_data'):
        # print(f)
        systerm_name = ''.join(f.split('_')[0]).lower()
        print(systerm_name)
        if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           'test_data')):
            os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'test_data'))
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           'prepare_data/{}_random_{}.csv'.format(systerm_name, all_data_nums))
        test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      'test_data/{}_test_data.csv'.format(systerm_name))
        print(test_data_path)
        print("_________造测试数据_______________")
        DATA = pd.read_csv(data_path).iloc[-test_size:, :]
        DATA = DATA.reset_index(drop=True)
        DATA.to_csv(test_data_path, index=False)


if __name__ == '__main__':

    # 配置所有数据集大小以及测试数据集大小
    all_data_nums = 300
    test_size=50
    process_300(all_data_nums)
    make_test_data(all_data_nums,test_size)
