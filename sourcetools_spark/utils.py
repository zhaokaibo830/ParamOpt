#!/user/bin/env python
# -*- coding: UTF-8 -*-
"""
-------------------------------
@Author : zspp
-------------------------------
@File : work_load_DDL9-- utils.py
@Software : PyCharm
@Time : 2022/6/10 20:13
-------------------------------
"""
from sobol_seq import i4_sobol_generate
import time
import random
import pandas as pd
import urllib
from urllib import request, parse
import sys
import socket
import random
import time

import numpy as np


# 随机采样
def random_sample_1(selected_params, sample_num):
    """
    对所有参数进行随机采样
    :param selected_params: selected_params字典形式
    :param sample_num: 采样数量
    :return: 采样结果
    """

    # 随机采样开始
    samples = []
    for _ in range(sample_num):
        samples.append([])
    for key in selected_params.keys():
        if selected_params[key][0] == 'int':
            for i in range(sample_num):
                samples[i].append(random.randint(selected_params[key][1][0] + 1, selected_params[key][1][1]))
        elif selected_params[key][0] == 'float':
            for i in range(sample_num):
                samples[i].append(random.uniform(selected_params[key][1][0] + 0.1, selected_params[key][1][1]))
        elif selected_params[key][0] == 'enum':
            for i in range(sample_num):
                samples[i].append(random.choice(selected_params[key][1]))
        else:
            continue

    # for i in range(sample_num):
    #     x = dict(zip(columns, samples[i]))
    #     y = get_performance(MODEL, x, SYSTERM_COLUMNS, max_X, max_Y)
    #     # y = random.random() #测试用
    #     samples[i].append(y)
    #     print("第{}条数据！！！\n".format(i))
    # columns.append(performance)

    df_samples = pd.DataFrame(samples, columns=list(selected_params.keys()))

    return df_samples

def random_sample_0(selected_params, sample_num, unimportant_feature_list):
    """
    不重要参数的采样为不重要参数随机采样，重要参数为默认值
    :param selected_params: selected_params字典形式
    :param sample_num: 采样数量
    :return: 采样结果
    """

    # 随机采样开始
    samples = []
    for _ in range(sample_num):
        samples.append([])
    for key in selected_params.keys():
        if key in unimportant_feature_list:
            if selected_params[key][0] == 'int':
                for i in range(sample_num):
                    samples[i].append(random.randint(selected_params[key][1][0] + 1, selected_params[key][1][1]))
            elif selected_params[key][0] == 'float':
                for i in range(sample_num):
                    samples[i].append(random.uniform(selected_params[key][1][0] + 0.1, selected_params[key][1][1]))
            elif selected_params[key][0] == 'enum':
                for i in range(sample_num):
                    samples[i].append(random.choice(selected_params[key][1]))
            else:
                continue
        else:
            for i in range(sample_num):
                samples[i].append(selected_params[key][2])

    # for i in range(sample_num):
    #     x = dict(zip(columns, samples[i]))
    #     y = get_performance(MODEL, x, SYSTERM_COLUMNS, max_X, max_Y)
    #     # y = random.random() #测试用
    #     samples[i].append(y)
    #     print("第{}条数据！！！\n".format(i))
    # columns.append(performance)

    df_samples = pd.DataFrame(samples, columns=list(selected_params.keys()))

    return df_samples



# 尽量全覆盖采样
def sobel_sample(selected_params, sobel_sample_number, important_feature_list):
    """
    进行sobel_sample
    :param selected_params: selected_params
    :param sobel_sample_number:100，采样数量
    :param important_feature_list: 重要的参数列表
    :return: 采样数据
    """

    sample_num = sobel_sample_number
    sample_index = i4_sobol_generate(len(important_feature_list), sample_num)

    samples = []
    for _ in range(sample_num):
        samples.append([])

    j = 0
    for key in selected_params.keys():
        if key in important_feature_list:
            if selected_params[key][0] == 'int':
                for i in range(sample_num):
                    value = int(sample_index[i][j] * (selected_params[key][1][1] - selected_params[key][1][0]) + \
                                selected_params[key][1][0])
                    samples[i].append(value)
            elif selected_params[key][0] == 'float':
                for i in range(sample_num):
                    value = sample_index[i][j] * (selected_params[key][1][1] - selected_params[key][1][0]) + \
                            selected_params[key][1][0]
                    samples[i].append(value)
            elif selected_params[key][0] == 'enum':
                for i in range(sample_num):
                    value = int(sample_index[i][j] * len(selected_params[key][1]))
                    samples[i].append(selected_params[key][1][value])
            else:
                continue
            j += 1
        else:
            for i in range(sample_num):
                samples[i].append(selected_params[key][2])

    df_samples = pd.DataFrame(samples, columns=list(selected_params.keys()))
    # print(df_samples)

    return df_samples


def get_performance(params):
    """
    接系统获取性能值
    :param params: {'param1': 10, 'param2': 's1', ...}
    :return: 性能值
    """

    headers = {"user-agent": "Mizilla/5.0"}
    value = parse.urlencode(params)
    # print(value)
    # data_ = value.encode('utf-8')
    # print(data_)
    # return -1

    # 请求url
    # http://47.104.101.50:8080/experiment/redis?     redis
    # http://47.104.101.50:9090/experiment/redis?     cassandra
    # http://47.104.101.50:8088/x264/exec?  x264
    # http://47.104.101.50:8081/jmeter?  tomcat
    req1 = request.Request('http://192.168.0.168:9002/api/spark/runWordcountForTime?%s' % value, headers=headers)  # 这样就能把参数带过去了
    print('http://192.168.0.168:9002/api/spark/runWordcountForTime?%s' % value)
    # 下面是获得响应
    i = 0
    # url_ = 'http://47.104.101.50:8081/jmeter'
    while True:
        try:
            # print(data_)
            f = request.urlopen(req1, timeout=360)

            # print(f)
            Data = f.read()
            data = Data.decode('utf-8')
            if data == "error":
                print("查不到，报errror，重新尝试\n")
                f.close()
                if i < 3:
                    i += 1
                    time.sleep(3)
                else:
                    return -1.0
            else:
                print(data)
                print("get_performance成功\n")
                f.close()
                break

        except Exception as e:
            print(e)
            i = i + 1
            if i < 3:
                time.sleep(3)
            else:
                return -1.0
    return (float)(data)


def process_data(data):
    """
    处理数据，对于枚举变量进行编码处理,按照代理模型的编码方式
    :return:
    """
    type_ = dict(data.dtypes)
    for i in type_.keys():
        if type_[i] == 'bool' or type_[i] == 'object':
            data[i] = data[i].astype('category').cat.codes
    return data


def get_performance_model(model, params, systerm_columns, max_X, max_Y, filename):
    # 根据代理模型获取性能
    """
    Args:
        max_Y，max_X: 模型归一化的参数
        model:代理模型
        params: 字典，要测量的一组参数
        systerm_columns: 列表;参数名称列表，对应模型的输入列


    Returns:
    """
    data = pd.DataFrame([params])
    data = data[systerm_columns]
    fake_data = pd.read_csv(filename)   # 这里加入多的假数据是为了编码，不影响预测结果，只去预测了要预测的
    fake_data = fake_data[systerm_columns]

    data = pd.concat([fake_data, data])
    data = process_data(data)
    data = np.divide(data.values, max_X)
    Y_pred = model.predict(data[-1:, :])
    Y_pred = max_Y * Y_pred

    return (float)(Y_pred)
