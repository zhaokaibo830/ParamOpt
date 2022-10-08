#!/user/bin/env python
# -*- coding: UTF-8 -*-
"""
-------------------------------
@Author : zspp
-------------------------------
@File : work_load_DDL9-- random_forest.py
@Software : PyCharm
@Time : 2022/6/10 12:26
-------------------------------
"""
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import os
from sklearn.preprocessing import StandardScaler
import json
from collections import *
from utils import *
import operator
# from tqdm import trange
from sklearn.metrics import mean_absolute_error
# from DeepPerf.mlp_sparse_model import *


# 随机森林预测模型对象
class random_forest(object):
    def __init__(self,task_name,target_model):

        # 加载相关配置参数
        self.confs = json.load(open('paras.json'), strict=False)

        # 参数名及其范围以及默认值
        self.selected_params = self.confs['common_params']['all_params']

        # 任务名称
        self.task_name = task_name

        # 性能列名
        self.performance = self.confs['common_params']['performance'].split('_')[0]

        # 性能值是极大还是极小
        self.min_or_max = self.confs['common_params']['performance'].split('_')[1]

        # 模型
        self.model = None

        # 模型保存路径
        if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           'target_output/{}'.format(self.task_name))):
            os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'target_output/{}'.format(self.task_name)))
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       'target_output/{}/random_forest.pickle'.format(self.task_name))

        # 特征重要性保存路径
        self.import_feature_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                'target_output/{}/random_forest_feature_importance.txt'.format(
                                                    self.task_name))

        # 初始数据集路径
        self.filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'data/target_data/{}.csv'.format(self.task_name))

        # 中间结果数据集路径
        self.temp_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          'target_output/{}/temp.csv'.format(self.task_name))

        # 数据列名
        self.columns = None
        self.fake_columns = None

        # 标准化模型
        self.scalar = None

        # 加载代理模型，不接系统的话
        self.MODEL =target_model
        # # 读取deepperf归一化变量
        # deepperf_scalar_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DeepPerf/result')
        # deepperf_scalar_name = 'result_{}_AutoML_veryrandom.txt'.format(self.task_name)
        # deepperf_scalar_path = os.path.join(deepperf_scalar_dir, deepperf_scalar_name)
        # file_read = open(deepperf_scalar_path, 'r', encoding='utf-8')
        # Params = file_read.read()
        # Params = dict(eval(Params))
        # self.max_X = Params['max_x']
        # self.max_Y = Params['max_y']
        # self.SYSTERM_COLUMNS = Params['SYSTERM_COLUMNS']

    def data_in(self, data):
        """
        数据编码,主要是对枚举变量编码
        :param data: 训练数据
        :return:
        """
        self.columns = list(data.columns)
        columns = self.columns[:-1]
        if self.min_or_max == 'max':
            data = data.sort_values(by=self.performance, ascending=False)
        else:
            data = data.sort_values(by=self.performance, ascending=True)
        data = data.drop(self.performance, 1)

        # 首先处理没有优先关系的enum参数，即string参数
        char = []  # enum的列名称
        enum_index = []  # enum的原始列索引
        for name in columns[:-1]:
            if self.selected_params[name][0] == 'string' or self.selected_params[name][0] == 'enum':
                char.append(name)
                enum_index.append(columns.index(name))

        enum_number = []  # 每个enum参数对应的独热编码的长度
        enum_book = {}  # 每个enum参数的内容，字典形式存储
        m = 0
        for c in char:
            i = enum_index[m]

            new_data = pd.DataFrame({c: self.selected_params[c][1]})  # 添加几行，为了更好全面编码
            data = data.append(new_data, ignore_index=True)

            enum_book[c] = list(pd.get_dummies(data[c]).columns)
            enum_data = pd.get_dummies(data[c], prefix=c)  # 独热编码后的变量

            data = data.drop(c, 1)

            enum_list = list(enum_data.columns)
            enum_number.append(len(enum_list))

            for k in range(len(enum_list)):
                data.insert(i + k, enum_list[k], enum_data[enum_list[k]])  # 将向量移动到原来枚举值的位置
            m = m + 1
            enum_index = [j + len(enum_data.columns) - 1 for j in enum_index]  # 更新enum_index

            data.drop(data.index[-len(self.selected_params[c][1]):], inplace=True)  # 删除前3行
            # print(enum_index)
        # print(enum_number)
        # print(data)
        # print(enum_book)
        self.fake_columns = list(data.columns)

        return data, enum_number, enum_book

    def data_out(self, data, enum_number, enum_book):
        """
        反向编码
        :param enum_number:
        :param enum_book:
        :return:
        """
        if data.empty:
            return pd.DataFrame(columns=self.columns[:-1])
        data.columns = [i for i in range(len(data.columns))]
        hang = data.iloc[:, 0].size

        # 将data的数据变为符合的类型格式
        m = 0  # 更改columns之后的columns的索引
        index_enum_number = 0  # enum_number的序号索引
        enum_list = []  # enum的列名称
        for name in self.columns[:-1]:
            if self.selected_params[name][0] == 'int':
                data[m] = data[m].astype(np.int64)
                m = m + 1
            elif self.selected_params[name][0] == 'float' or self.selected_params[name][0] == 'double':
                m = m + 1
                continue
            else:
                enum_list.append(name)
                for i in range(enum_number[index_enum_number]):
                    data[m + i] = data[m + i].round().astype(np.int64)
                m = m + enum_number[index_enum_number]
                index_enum_number = index_enum_number + 1
        # print(data)

        # 在数据尾部加入enum的原始列名称
        for i in enum_list:
            data_temp = pd.DataFrame(columns=[i])
            data = pd.concat([data, data_temp], 1)

        for index in range(hang):
            # 将向量转换成枚举值
            # print(data.iloc[[index]].values[0])
            for k in range(len(enum_list)):

                name_index = self.columns.index(enum_list[k])  # enum变量的原始列索引
                if k != 0:
                    for a in range(k):
                        name_index = name_index + enum_number[a] - 1  # 变为enum改变后的索引
                # print(name_index)

                flag = True
                true_index = False
                first_index = -1
                for i in range(enum_number[k]):

                    for j in range(i, enum_number[k]):
                        if i == j:

                            if round(data.loc[[index]].values[0][name_index + j]) == 1:
                                first_index = j
                                continue
                            else:
                                if j == enum_number[k] - 1:
                                    flag = False
                                break
                        else:
                            if round(data.loc[[index]].values[0][name_index + j]) == 0:

                                if j == enum_number[k] - 1:
                                    true_index = True
                                    break
                                else:
                                    continue
                            else:
                                flag = False
                                break

                    if flag == False:
                        break
                    if true_index == True:
                        break
                if flag == False:
                    data.drop([index], inplace=True)
                    break
                if (first_index + 1) != 0 and flag == True:
                    data.loc[index, enum_list[k]] = enum_book[enum_list[k]][first_index]
        # print(data)

        # 删去枚举值所在的列
        number = 0
        for i in range(len(enum_list)):
            name_index = number + self.columns.index(enum_list[i])

            for j in range(enum_number[i]):
                data = data.drop([name_index + j], 1)

            number = number + enum_number[i] - 1
        # print(data)

        # 移动最后含枚举值的几列,对列进行重命名
        for i in range(len(enum_list)):
            orgin_index = self.columns.index(enum_list[i])
            data.insert(orgin_index, enum_list[i], data.pop(enum_list[i]))
        # print(data)

        col = {}
        for (key, value) in zip(list(data.columns), self.columns[:-1]):
            col[key] = value
        # print(col)
        data.rename(columns=col, inplace=True)
        # print(data)
        return data

    def data_range(self, data, enum_number):
        """
        判断采样的返回的结果是否再参数的范围内
        :param data:
        :param enum_number:
        :return:
        """
        data = data.values
        MAX_BOUND = []
        MIN_BOUND = []
        m = 0
        for name in self.columns[:-1]:
            if self.selected_params[name][0] == 'enum' or self.selected_params[name][0] == 'string':
                for i in range(enum_number[m]):
                    MIN_BOUND.append(0)
                    MAX_BOUND.append(1)
                m = m + 1

            else:
                MIN_BOUND.append(self.selected_params[name][1][0])
                MAX_BOUND.append(self.selected_params[name][1][1])
        MIN_BOUND = [i - 1 for i in MIN_BOUND]
        MAX_BOUND = [i + 1 for i in MAX_BOUND]

        result = []
        for each in data:
            flag = True
            for j in range(len(MIN_BOUND)):
                if each[j] <= MIN_BOUND[j] or each[j] >= MAX_BOUND[j]:
                    flag = False
                    break
            if flag:
                result.append(each)

        return pd.DataFrame(np.array(result))

    def one_step_train(self, data):
        """
        传入训练数据训练随机森林预测模型
        :param data: 训练数据
        :return:
        """
        # 原始数据编码，没有采用sklearn自带的编码，是因为要接入系统考虑，所以用跟之前benchmark实验一致的编码手段
        data_y = data.iloc[:, -1].values
        data_x, enum_number, enum_book = self.data_in(data)
        # print(data_x)
        # print(data_y)

        # 标准化
        self.scalar = StandardScaler()
        data_x = self.scalar.fit_transform(data_x.astype(float))
        # print(data_x)

        # 随机森林训练
        self.model = RandomForestRegressor()
        self.model.fit(data_x, data_y)

        # 保存训练好的随机森林模型
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)

    def target_train(self):

        origin_data = pd.read_csv(self.filename)
        unimportant_feature_sample_nums = self.confs['sample_params']['unimportant_feature_sample_nums']  # 初始不重要参数采样数量
        temp_data = pd.DataFrame(columns=origin_data.columns)  # 中间结果保存结果集
        # 迭代采样+与系统交互
        for i in range(self.confs['sample_params']['sample_epoch']):
            self.one_step_train(origin_data)
            # 获取重要性特征
            sort_features = self.get_sort_feature()
            # print(sort_features)

            # 分别采样重要参数、不重要的参数
            # 不重要的参数随机采样
            sample_data_1 = random_sample(self.selected_params,
                                          unimportant_feature_sample_nums, sort_features[-(
                        len(self.selected_params) - self.confs['sample_params']['important_feature_nums']):])

            # 重要的参数覆盖采样
            sample_data_2 = sobel_sample(self.selected_params,
                                         self.confs['sample_params']['important_feature_sample_nums'],
                                         sort_features[:self.confs['sample_params']['important_feature_nums']])
            # print(len(sample_data_2))
            # 合并采样结果
            sample_data = pd.concat([sample_data_1, sample_data_2], axis=0)
            sample_data = sample_data.reset_index(drop=True)

            # print(sample_data.columns)
            # 使用训练好的模型进行预测
            predict = self.test_(sample_data)
            # print(predict)
            if self.min_or_max == 'max':
                max_index = np.argmax(predict)
            else:
                max_index = np.argmin(predict)
            sample_data.drop(self.performance, axis=1, inplace=True)

            # 找到性能值最优的那一个，接入到实际的系统跑一下
            top_one = dict(sample_data.iloc[max_index, :])
            # print(top_one)
            # perf = get_performance(top_one)

            # 没有实际系统，接代理模型得到性能值
            target_model_temp = self.MODEL
            x=top_one
            j = 0
            while j < len(target_model_temp):
                if target_model_temp[j] == "o":
                    k = j + 1
                    while k < len(target_model_temp) and target_model_temp[k].isdigit():
                        k = k + 1
                    # print(target_model_temp[j:k])
                    index = target_model_temp[j:k]
                    temp = target_model_temp[:j] + str(x[index])
                    if k < len(target_model_temp):
                        target_model_temp = temp + target_model_temp[k:]
                    else:
                        target_model_temp = temp
                    j = j + len(str(x[index]))
                else:
                    j = j + 1
            perf = eval(target_model_temp)

            # print(perf)

            print("第{}轮：真实最优参数为{},对应的性能值为{}".format(i + 1, top_one, perf))
            top_one[self.performance] = perf

            # 合并这个结果到初始数据集
            one_data = pd.DataFrame([top_one])
            origin_data = pd.concat([origin_data, one_data[list(origin_data.columns)]], axis=0)
            origin_data = origin_data.reset_index(drop=True)

            # 合并保存中间接系统或者代理模型的结果
            temp_data = pd.concat([temp_data, one_data], axis=0)

            # 随机采样数量衰减
            # 动态根据公式调整不重要参数的数量
            theta = -50 / math.log(0.5)
            unimportant_feature_sample_nums = int(unimportant_feature_sample_nums * math.exp(
                -math.pow(max(0, i - 20), 2) / (2 * theta * theta)))

            # self.test_test_data()

        temp_data.to_csv(self.temp_filename, index=False)

    def get_sort_feature(self):
        """
        获取重要的参数特征
        :return:
        """
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        # 获取权重以及其对应的权重
        feature_weight_dict = defaultdict(int)
        for i in range(len(self.fake_columns)):
            if "_" in self.fake_columns[indices[i]]:
                feature_weight_dict[self.fake_columns[indices[i]].split('_')[0]] = max(importances[indices[i]],
                                                                                       feature_weight_dict[
                                                                                           self.fake_columns[
                                                                                               indices[i]].split(
                                                                                               '_')[0]])
            else:
                feature_weight_dict[self.fake_columns[indices[i]]] = importances[indices[i]]

        feature_weight_dict = dict(sorted(feature_weight_dict.items(), key=operator.itemgetter(1), reverse=True))
        # 保存特征及其对应的权重
        with open(self.import_feature_path, 'w') as f:
            f.write(str(feature_weight_dict))

        # 返回排序后的特征列表
        return list(feature_weight_dict.keys())

    def test_(self, data):
        data[self.performance] = 0
        data, _, _ = self.data_in(data)
        data = self.scalar.transform(data)
        return self.model.predict(data)

    def test_test_data(self):
        test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      'test_data/{}_test_data.csv'.format(self.task_name))

        test_data = pd.read_csv(test_data_path)
        y_test = test_data[self.performance].values.copy()
        test_data[self.performance] = 0
        data, _, _ = self.data_in(test_data)
        data = self.scalar.transform(data)
        predict = self.model.predict(data)

        # print(predict.shape)
        # print(y_test.shape)
        mae = np.abs(y_test - predict)
        rmse = np.divide(mae, y_test)
        rmse = sum(rmse)/len(rmse)
        # print(rmse.shape)
        print("测试集的map损失为：{}".format(rmse))

    def cal_Similarity(self,sourcetask_models_path, origin_data):
        """
        计算每个原任务和目标任务之间的相似度
        :param sourcetask_models: 原模型列表
        :param target_X:
        :param target_Y:
        :return:
        """
        target_X=origin_data.iloc[:,:-1].values
        target_Y = origin_data.iloc[:, -1].values

        for sourcetask_name in os.listdir(sourcetask_models_path):
            # print(sourcetask_name)
            with open(sourcetask_models_path+"/{}/random_forest.pickle".format(sourcetask_name), 'rb') as f:
                sourcetask_index = int(sourcetask_name.split('_')[1].split(".")[0])

                self.sourcetask_models[sourcetask_index-1] = pickle.load(f)

        similarity_list = []
        for sourcetask_model in self.sourcetask_models:
            rank_loss = 0
            for j in range(len(target_X)):
                for k in range(j + 1, len(target_X)):
                    target_X_j = sourcetask_model.predict([target_X[j]])
                    target_X_k = sourcetask_model.predict([target_X[k]])
                    if (target_X_j[0] < target_X_k[0]) ^ (target_Y[j] < target_Y[k]):
                        rank_loss += 1
            similarity_list.append(2 * rank_loss / (len(target_Y) / (len(target_Y) - 1)))
        return similarity_list

    def choose_source(self,slected_k_source,similarity_list,sourcetask_svm_classifier):
        """
        选择k个原任务
        :param similarity_list:
        :return: 返回选择的原任务下标
        """
        similarity_list_sum=sum(similarity_list)
        p_list=[v/similarity_list_sum for v in similarity_list]
        if sum(p_list)!= 1:
            p_list[-1]=1-sum(p_list[:-1])
        selected_svm_classifier=np.random.choice(sourcetask_svm_classifier,slected_k_source,replace=False,p=p_list)
        return selected_svm_classifier

    def condense_sampledata(self, selected_svm_classifier, sample_data,slected_k_source):
        condenseed_sample_data=[]
        for one_data in sample_data.values:
            count=0
            for clff in selected_svm_classifier:
                if clff.predict([one_data])==1:
                    count += 1
            if count>=slected_k_source//2:
                condenseed_sample_data.append(one_data)
        return pd.DataFrame(condenseed_sample_data, columns=sample_data.columns)

    def get_sourcetask_svm_clf(self, clff_path):
        sourcetask_svm_classifier=[None]*self.sourcetask_num
        for name in os.listdir(clff_path):
            with open(clff_path+name,"rb") as f:
                index=int(name.split("_")[1].split(".")[0])
                sourcetask_svm_classifier[index-1]=pickle.load(f)
        return sourcetask_svm_classifier

