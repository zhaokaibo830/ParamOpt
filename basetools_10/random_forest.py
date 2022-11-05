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
from scipy.stats import norm
from tensorboardX import SummaryWriter
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
from tqdm import trange
from concurrent.futures import ThreadPoolExecutor
from sklearn import svm
from sklearn.metrics import mean_absolute_error
# from DeepPerf.mlp_sparse_model import *
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


# 随机森林预测模型对象
class random_forest(object):
    def __init__(self,task_type,task_name,target_model):

        # 加载相关配置参数
        self.confs = json.load(open('paras.json'), strict=False)

        # 参数名及其范围以及默认值
        self.selected_params = self.confs['common_params']['all_params']

        # 任务名称
        self.task_name = task_name

        # 采集函数是否选择EI，1表示选择，0表示没被选择
        self.is_EI = self.confs['is_EI']
        self.target_output_name = "_isEI" if self.is_EI else "_noEI"

        # 1表示不重要参数的采样为对所有参数进行随机采样，0表示不重要参数的采样为不重要参数随机采样，重要参数为默认值
        self.select_unimportant_parameter_samples=self.confs['select_unimportant_parameter_samples']
        self.select_unimportant_parameter_samples_name = "_sups1" if self.select_unimportant_parameter_samples else "_sups0"

        # 候选补齐方式，0表示不补齐，1表示样本太少在投票一直的范围内随机采样
        self.candidate_supplement=self.confs['candidate_supplement']
        self.candidate_supplement_name="_cs1" if self.candidate_supplement else "_cs0"

        # 候选补齐门限
        self.candidate_supplement_threshold = self.confs['candidate_completion_threshold']

        # 源任务选择方式，0表示按照相似度计算得到的概率获取，概率越大被选择的概率越大，1表示按照相似度大小取，取相似度最大的几个
        self.sourcetasks_get_method=self.confs['sourcetasks_get_method']
        self.sourcetasks_get_method_name="_sget1" if self.sourcetasks_get_method else "_sget0"



        if task_type=="target":
            # 选择原任务的个数
            self.slected_k_source=self.confs['slected_k_source']

            # 原任务数量
            self.sourcetask_num=self.confs['sourcetask_num']

            # 原任务模型
            self.sourcetask_models = [None] * self.sourcetask_num

            # 原任务分类器需要的参数
            self.alpha_max=self.confs['alpha_max']
            self.alpha_min=self.confs['alpha_min']

            # 相似度列表
            self.similarity_list=[]

        # 性能列名
        self.performance = self.confs['common_params']['performance'].split('_')[0]

        # 性能值是极大还是极小
        self.min_or_max = self.confs['common_params']['performance'].split('_')[1]

        # 模型
        self.model = None


        if task_type=="source":

            # 模型保存路径
            if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               'output{}{}{}{}/source/{}'.format(self.target_output_name,
                                                                             self.select_unimportant_parameter_samples_name,
                                                                               self.candidate_supplement_name,
                                                                            self.sourcetasks_get_method_name,
                                                                             self.task_name))):
                os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         'output{}{}{}{}/source/{}'.format(self.target_output_name,
                                                                       self.select_unimportant_parameter_samples_name,
                                                                         self.candidate_supplement_name,
                                                                           self.sourcetasks_get_method_name,
                                                                       self.task_name)))
            self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           'output{}{}{}{}/source/{}/random_forest.pickle'.format(self.target_output_name,
                                                                                              self.select_unimportant_parameter_samples_name,
                                                                                                self.candidate_supplement_name,
                                                                                                  self.sourcetasks_get_method_name,
                                                                                              self.task_name))

            # 特征重要性保存路径
            self.import_feature_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    'output{}{}{}{}/source/{}/random_forest_feature_importance.txt'.format(self.target_output_name,
                                                                                                                       self.select_unimportant_parameter_samples_name,
                                                                                                                         self.candidate_supplement_name,
                                                                                                                           self.sourcetasks_get_method_name,
                                                                                                                       self.task_name))

            # 初始数据集路径
            self.filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         'data/source_data/{}.csv'.format(self.task_name))

            # 数据保存位置
            self.save_filesname = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               'output{}{}{}{}/source/{}/'.format(self.target_output_name,
                                                                              self.select_unimportant_parameter_samples_name,
                                                                                self.candidate_supplement_name,
                                                                                  self.sourcetasks_get_method_name,
                                                                              self.task_name))
        else:
            if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               'output{}{}{}{}/target'.format(self.target_output_name,
                                                                            self.select_unimportant_parameter_samples_name,
                                                                            self.candidate_supplement_name,
                                                                              self.sourcetasks_get_method_name))):
                os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         'output{}{}{}{}/target/'.format(self.target_output_name,
                                                                       self.select_unimportant_parameter_samples_name,
                                                                       self.candidate_supplement_name,
                                                                         self.sourcetasks_get_method_name)))
            self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           'output{}{}{}{}/target/random_forest.pickle'.format(self.target_output_name,
                                                                                             self.select_unimportant_parameter_samples_name,
                                                                                             self.candidate_supplement_name,
                                                                                               self.sourcetasks_get_method_name))

            # 分类器可视化保存位置
            if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               'output{}{}{}{}/target/vis_cls'.format(self.target_output_name,
                                                                                    self.select_unimportant_parameter_samples_name,
                                                                                    self.candidate_supplement_name,
                                                                                      self.sourcetasks_get_method_name))):
                os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         'output{}{}{}{}/target/vis_cls'.format(self.target_output_name,
                                                                              self.select_unimportant_parameter_samples_name,
                                                                              self.candidate_supplement_name,
                                                                                self.sourcetasks_get_method_name)))
            self.vis_cls_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           'output{}{}{}{}/target/vis_cls'.format(self.target_output_name,
                                                                                self.select_unimportant_parameter_samples_name,
                                                                                self.candidate_supplement_name,
                                                                                  self.sourcetasks_get_method_name))

            # 特征重要性保存路径
            self.import_feature_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    'output{}{}{}{}/target/random_forest_feature_importance.txt'.format(
                                                        self.target_output_name,
                                                        self.select_unimportant_parameter_samples_name,
                                                        self.candidate_supplement_name,
                                                    self.sourcetasks_get_method_name,))

            # 初始数据集路径
            self.filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         'data/target_data/{}.csv'.format(self.task_name))

            # 数据保存位置
            self.save_filesname = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              'output{}{}{}{}/target/'.format(self.target_output_name,
                                                                            self.select_unimportant_parameter_samples_name,
                                                                            self.candidate_supplement_name,
                                                                              self.sourcetasks_get_method_name))


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
        data_x = data.iloc[:, :-1].values
        self.fake_columns = list(data.iloc[:, :-1].columns)
        # data_x, enum_number, enum_book = self.data_in(data)

        # print("data_y")
        # print(data_y)

        # 标准化
        # self.scalar = StandardScaler()
        # data_x = self.scalar.fit_transform(data_x.astype(float))
        # print("data_x")
        # print(data_x.values)

        # 随机森林训练
        self.model = RandomForestRegressor()
        self.model.fit(data_x, data_y)

        # 保存训练好的随机森林模型
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)

    def target_train(self):

        origin_data = pd.read_csv(self.filename)
        unimportant_feature_sample_nums = self.confs['sample_params']['unimportant_feature_sample_nums']  # 初始不重要参数采样数量
        important_feature_sample_num = self.confs['sample_params']['important_feature_sample_num']  # 初始重要参数采样数量
        sample_step_change = self.confs['sample_params']['sample_step_change']  # 每步重要参数采样增加个数
        temp_data_real_perf_min = pd.DataFrame(columns=origin_data.columns)  # 真实中间结果保存结果集
        temp_data_forest_perf_min = pd.DataFrame(columns=origin_data.columns)  # 随机森林中间结果保存结果集
        eta = max(origin_data.iloc[:, -1].values) if self.min_or_max == 'max'else min(origin_data.iloc[:, -1].values) # eta是当前最优值
        #  eta_index表示最小值获取的代数或者初始样本的位置，如果小于表示最小值是在初始样本得到的
        eta_index=-np.argmax(origin_data.iloc[:, -1].values) if self.min_or_max == 'max'else -np.argmin(origin_data.iloc[:, -1].values)

        every_epoch_remain_sample_num = []  # 每一轮剩余的样本数
        every_epoch_remain_sample_ratio=[]  # 每一轮剩余的样本数所占比例
        similarity_save = pd.DataFrame(columns=["sourcetask_"+str(i) for i in range(1,self.sourcetask_num+1)])  #相似度保存位置
        # mre_sum=[0]   # mre_sum为前面所有轮次随机森林和真实性能的mre误差之和
        mre=[]  #mre为每一轮随机森林和真实性能的mre误差
        selectd_important_features_save = []  # 保存每轮选择的重要参数

        condenseed_sample_data_iszero_iter=[]

        writer = SummaryWriter(self.save_filesname+"runs/")
        # 迭代采样+与系统交互
        for i in trange(self.confs['sample_params']['sample_epoch']):
            # print("______________")
            # print(i)
            self.one_step_train(origin_data)
            # 获取重要性特征
            sort_features = self.get_sort_feature()
            # print(sort_features)
            # 保存当前轮的重要特征
            selectd_important_features_save.append(sort_features[:self.confs['sample_params']['important_feature_nums']])
            # 计算相似性列表
            self.similarity_list=self.cal_Similarity("./output{}{}{}{}/source/".format(self.target_output_name,
                                                                                   self.select_unimportant_parameter_samples_name,
                                                                                     self.candidate_supplement_name,
                                                                                       self.sourcetasks_get_method_name),origin_data)
            # print(self.similarity_list)

            # 保存相似度
            similarity_one_iter = pd.DataFrame(data=[self.similarity_list], columns=similarity_save.columns)
            similarity_save=pd.concat([similarity_save,similarity_one_iter],axis=0)


            # 选择原任务,得到原任务对应的索引
            self.selected_svm_index=self.choose_source(self.slected_k_source,self.similarity_list,self.sourcetask_num,self.sourcetasks_get_method)

            # 获取原任务分类器
            self.svm_classifier=self.get_selected_sourcetask_svm_clf(list(range(1,self.sourcetask_num+1)))
            self.selected_svm_classifier = np.array(self.svm_classifier)[self.selected_svm_index]

            # 分别采样重要参数、不重要的参数
            # 不重要的参数随机采样,
            if self.select_unimportant_parameter_samples:
                # 1表示不重要参数的采样为对所有参数进行随机采样，
                sample_data_1 = random_sample_1(self.selected_params,
                                              unimportant_feature_sample_nums)
            else:
                # 表示不重要参数的采样为不重要参数随机采样，重要参数为默认值
                sample_data_1 = random_sample_0(self.selected_params,
                                              unimportant_feature_sample_nums, sort_features[-(
                            len(self.selected_params) - self.confs['sample_params']['important_feature_nums']):])


            # 重要的参数覆盖采样
            sample_data_2 = sobel_sample(self.selected_params,
                                         important_feature_sample_num,
                                         sort_features[:self.confs['sample_params']['important_feature_nums']])

            # 合并采样结果
            sample_data = pd.concat([sample_data_1, sample_data_2], axis=0)
            sample_data = sample_data.reset_index(drop=True)


            # 获取压缩后得到采样数据
            condenseed_sample_data,discard_data=self.condense_sampledata(self.selected_svm_classifier,sample_data,self.slected_k_source)

            # 把此轮剩余的样本数加到every_epoch_remain_sample_num末尾
            every_epoch_remain_sample_num.append(len(condenseed_sample_data.values))
            every_epoch_remain_sample_ratio.append(len(condenseed_sample_data.values) / len(sample_data.values))

            if self.candidate_supplement==0:
                if len(condenseed_sample_data.iloc[:,:].values) == 0:
                    condenseed_sample_data = sample_data.iloc[:2, :]
            elif self.candidate_supplement==1:
                m=0
                while len(condenseed_sample_data.iloc[:,:].values)<self.candidate_supplement_threshold and m<50:
                    candidate_sample_data=random_sample_1(self.selected_params,important_feature_sample_num)
                    candidate_sample_data,_=self.condense_sampledata(self.selected_svm_classifier,candidate_sample_data,self.slected_k_source)
                    condenseed_sample_data=pd.concat([condenseed_sample_data, candidate_sample_data], axis=0)
                    condenseed_sample_data = condenseed_sample_data.reset_index(drop=True)
                    m += 1
                if len(condenseed_sample_data.iloc[:,:].values)==0:
                    condenseed_sample_data_iszero_iter.append(i)
                    condenseed_sample_data = sample_data.iloc[:, :]



            # 可视化分类器
            # self.vis_cls(self.selected_svm_index,self.svm_classifier, self.vis_cls_path, self.confs, i + 1, condenseed_sample_data, discard_data,self.similarity_list)


            if self.is_EI:
                # 使用训练好的模型进行预测，获取每个样本对应的EI值
                ei = self.get_ei(self.model, eta, condenseed_sample_data)
                max_index = np.argmax(ei)
            else:
                # 采集函数使用起初基础调优器给的
                predict = self.test_(condenseed_sample_data)
                # print(predict)
                if self.min_or_max == 'max':
                    max_index = np.argmax(predict)
                else:
                    max_index = np.argmin(predict)
                # sample_data.drop(self.performance, axis=1, inplace=True)

            # 找到性能值最优的那一个，接入到实际的系统跑一下
            top_one = dict(condenseed_sample_data.iloc[max_index, :])
            # print(top_one)
            # perf = get_performance(top_one)

            # 没有实际系统，接代理模型得到性能值
            perf = self.get_agentmodel_value(self.MODEL, top_one)

            # 更新eta
            if self.min_or_max == 'max':
                if eta < perf:
                    eta_index = i + 1
                eta = max(eta, perf)
            else:
                if eta > perf:
                    eta_index = i + 1
                eta = min(eta, perf)

            # print("第{}轮：真实最优参数为{},对应的性能值为{}".format(i + 1, top_one, perf))
            top_one[self.performance] = perf

            # 合并这个结果到初始数据集
            one_data = pd.DataFrame([top_one])
            origin_data = pd.concat([origin_data, one_data[list(origin_data.columns)]], axis=0)
            origin_data = origin_data.reset_index(drop=True)

            # 合并保存真实系统或者代理模型的结果
            temp_data_real_perf_min = pd.concat([temp_data_real_perf_min, one_data], axis=0)

            top_one[self.performance] = self.model.predict([condenseed_sample_data.iloc[max_index, :]])[0]
            one_data = pd.DataFrame([top_one])
            # 合并保存随机森林的结果
            temp_data_forest_perf_min = pd.concat([temp_data_forest_perf_min, one_data], axis=0)

            # 计算mre
            one_mre=self.cal_mre(self.MODEL,condenseed_sample_data)
            mre.append(one_mre)


            # 重要参数采样数量递增
            important_feature_sample_num = important_feature_sample_num + sample_step_change

            # 随机采样数量衰减
            # 动态根据公式调整不重要参数的数量
            theta = -50 / math.log(0.5)
            unimportant_feature_sample_nums = int(unimportant_feature_sample_nums * math.exp(
                -math.pow(max(0, i - 20), 2) / (2 * theta * theta)))

            writer.add_scalar('perf_min_forest', self.model.predict([condenseed_sample_data.iloc[max_index, :]])[0], global_step=i)
            writer.add_scalar('perf_min', perf, global_step=i)
            writer.add_scalar('mre', mre[-1], global_step=i)
            np.savetxt(self.save_filesname + 'every_epoch_remain_sample_ratio.txt', every_epoch_remain_sample_ratio)

            similarity_save.to_csv(self.save_filesname+"similarity.csv", index=False)
            np.savetxt(self.save_filesname+'every_epoch_remain_sample_num.txt', every_epoch_remain_sample_num)
            np.savetxt(self.save_filesname+'every_epoch_remain_sample_ratio.txt', every_epoch_remain_sample_ratio)
            temp_data_real_perf_min.to_csv(self.save_filesname + "temp_data_real_perf_min.csv", index=False)
            temp_data_forest_perf_min.to_csv(self.save_filesname + "temp_data_forest_perf_min.csv", index=False)
        np.savetxt(self.save_filesname + 'mre.txt', mre)
        # np.savetxt(self.save_filesname + 'mre_sum.txt', mre_sum)
        np.savetxt(self.save_filesname + 'condenseed_sample_data_iszero_iter.txt', condenseed_sample_data_iszero_iter)
        with open(self.save_filesname + 'optimal_value.json', "w") as f:
            json.dump({"index":str(eta_index),"opt_perf":float(eta)}, f)
        np.savetxt(self.save_filesname + 'selectd_important_features_save.txt',np.array(selectd_important_features_save), fmt="%s")
        # self.vis_tempfile(self.task_name,self.save_filesname+"temp_data_real_perf_min.csv",self.save_filesname)


    def source_train(self):

        origin_data = pd.read_csv(self.filename)
        unimportant_feature_sample_nums = self.confs['sample_params']['unimportant_feature_sample_nums']  # 初始不重要参数采样数量
        important_feature_sample_num = self.confs['sample_params']['important_feature_sample_num']  # 初始重要参数采样数量
        sample_step_change = self.confs['sample_params']['sample_step_change']  # 每步重要参数采样增加个数
        temp_data_real_perf_min = pd.DataFrame(columns=origin_data.columns)  # 真实中间结果保存结果集
        temp_data_forest_perf_min = pd.DataFrame(columns=origin_data.columns)  # 随机森林中间结果保存结果集
        eta = max(origin_data.iloc[:, -1].values) if self.min_or_max == 'max'else min(origin_data.iloc[:, -1].values) # eta是当前最优值
        #  eta_index表示最小值获取的代数或者初始样本的位置，如果小于表示最小值是在初始样本得到的
        eta_index=-np.argmax(origin_data.iloc[:, -1].values) if self.min_or_max == 'max'else -np.argmin(origin_data.iloc[:, -1].values)
        # mre_sum = [0]  # mre_sum为前面所有轮次随机森林和真实性能的mre误差之和
        # mre = [0]  # mre为每一轮随机森林和真实性能的mre误差
        selectd_important_features_save = []  # 保存每轮选择的重要参数

        writer = SummaryWriter(self.save_filesname+"runs/")
        # 迭代采样+与系统交互
        for i in range(self.confs['sample_params']['sample_epoch']):
            self.one_step_train(origin_data)
            # 获取重要性特征
            sort_features = self.get_sort_feature()
            # print(sort_features)
            selectd_important_features_save.append(sort_features[:self.confs['sample_params']['important_feature_nums']])

            # 分别采样重要参数、不重要的参数
            # 不重要的参数随机采样,
            if self.select_unimportant_parameter_samples:
                # 1表示不重要参数的采样为对所有参数进行随机采样，
                sample_data_1 = random_sample_1(self.selected_params,
                                              unimportant_feature_sample_nums)
            else:
                # 表示不重要参数的采样为不重要参数随机采样，重要参数为默认值
                sample_data_1 = random_sample_0(self.selected_params,
                                              unimportant_feature_sample_nums, sort_features[-(
                            len(self.selected_params) - self.confs['sample_params']['important_feature_nums']):])

            # 重要的参数覆盖采样
            sample_data_2 = sobel_sample(self.selected_params,
                                         important_feature_sample_num,
                                         sort_features[:self.confs['sample_params']['important_feature_nums']])
            # print(len(sample_data_2))
            # 合并采样结果
            sample_data = pd.concat([sample_data_1, sample_data_2], axis=0)
            sample_data = sample_data.reset_index(drop=True)

            # print(sample_data.columns)
            if self.is_EI:
                # 使用训练好的模型进行预测，获取每个样本对应的EI值
                ei = self.get_ei(self.model, eta, sample_data)
                max_index = np.argmax(ei)
            else:
                # 采集函数使用起初基础调优器给的
                predict = self.test_(sample_data)
                # print(predict)
                if self.min_or_max == 'max':
                    max_index = np.argmax(predict)
                else:
                    max_index = np.argmin(predict)
                # sample_data.drop(self.performance, axis=1, inplace=True)

            # 找到性能值最优的那一个，接入到实际的系统跑一下
            top_one = dict(sample_data.iloc[max_index, :])
            # print(top_one)
            # perf = get_performance(top_one)

            # 没有实际系统，接代理模型得到性能值
            perf = self.get_agentmodel_value(self.MODEL, top_one)

            # 更新eta
            if self.min_or_max == 'max':
                if eta <perf:
                    eta_index=i+1
                eta = max(eta, perf)
            else:
                if eta > perf:
                    eta_index=i+1
                eta = min(eta, perf)

            # print(perf)

            # print("第{}轮：真实最优参数为{},对应的性能值为{}".format(i + 1, top_one, perf))
            top_one[self.performance] = perf

            # 合并这个结果到初始数据集
            one_data = pd.DataFrame([top_one])
            origin_data = pd.concat([origin_data, one_data[list(origin_data.columns)]], axis=0)
            origin_data = origin_data.reset_index(drop=True)
            # 合并保存真实中间接系统或者代理模型的结果
            temp_data_real_perf_min = pd.concat([temp_data_real_perf_min, one_data], axis=0)

            top_one[self.performance] = self.model.predict([sample_data.iloc[max_index, :]])[0]
            one_data = pd.DataFrame([top_one])
            # 合并保存中间接系统或者代理模型的结果
            temp_data_forest_perf_min = pd.concat([temp_data_forest_perf_min, one_data], axis=0)

            # 计算mre
            # mre_sum.append(
            #     mre_sum[-1] + np.fabs(perf - self.model.predict([sample_data.iloc[max_index, :]])[0]) * 100 / np.fabs(perf))
            # mre.append(mre_sum[-1] / (i + 1))


            # 重要参数采样数量递增
            important_feature_sample_num = important_feature_sample_num + sample_step_change

            # 随机采样数量衰减
            # 动态根据公式调整不重要参数的数量
            theta = -50 / math.log(0.5)
            unimportant_feature_sample_nums = int(unimportant_feature_sample_nums * math.exp(
                -math.pow(max(0, i - 20), 2) / (2 * theta * theta)))

            # self.test_test_data()
            writer.add_scalar('perf_min_forest', self.model.predict([sample_data.iloc[max_index, :]])[0], global_step=i)
            writer.add_scalar('perf_min', perf, global_step=i)
            # writer.add_scalar('mre', mre[-1], global_step=i)
        # np.savetxt(self.save_filesname + 'mre.txt', mre)
        # np.savetxt(self.save_filesname + 'mre_sum.txt', mre_sum)
        with open(self.save_filesname + 'optimal_value.json', "w") as f:
            json.dump({"index":str(eta_index),"opt_perf":float(eta)}, f)
        np.savetxt(self.save_filesname + 'selectd_important_features_save.txt',np.array(selectd_important_features_save), fmt="%s")
        temp_data_real_perf_min.to_csv(self.save_filesname + "temp_data_real_perf_min.csv", index=False)
        temp_data_forest_perf_min.to_csv(self.save_filesname + "temp_data_forest_perf_min.csv", index=False)
        # self.vis_tempfile(self.task_name,self.save_filesname + "temp_data_forest_perf_min.csv", self.save_filesname)

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
        with open(self.import_feature_path, 'a') as f:
            f.write(str(feature_weight_dict)+"\n")

        # 返回排序后的特征列表
        return list(feature_weight_dict.keys())

    def test_(self, data):
        # data[self.performance] = 0
        # data, _, _ = self.data_in(data)
        # data = self.scalar.transform(data)
        return self.model.predict(data.values)

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

        similarity_list=list(map(self.cal_one_Similarity,self.sourcetask_models,[target_X]*len(self.sourcetask_models),[target_Y]*len(self.sourcetask_models)))

        return similarity_list

    def cal_one_Similarity(self,sourcetask_model,target_X,target_Y):
        source_M=list(map(self.cal_Mxi,[sourcetask_model]*len(target_X),target_X))
        # print(source_M)
        source_M_col=np.array(source_M).reshape(len(target_Y),1)
        source_M_row=source_M_col.T
        target_Y_col=np.array(target_Y).reshape(len(target_Y),1)
        target_Y_row=target_Y_col.T
        temp=np.logical_not(np.logical_xor(source_M_col<source_M_row,target_Y_col<target_Y_row))
        rank_loss=np.sum(temp[np.triu_indices(len(target_Y),1)])
        return 2 * rank_loss / (len(target_Y) * (len(target_Y) - 1))

    def cal_Mxi(self,sourcetask_model,target_X_i):
        return sourcetask_model.predict([target_X_i])[0]

    def choose_source(self,slected_k_source,similarity_list,sourcetask_num,sourcetasks_get_method):
        """
        选择k个原任务, 源任务选择方式，sourcetasks_get_method为0表示按照相似度计算得到的概率获取，
        概率越大被选择的概率越大，为1表示按照相似度大小取，取相似度最大的几个
        :param similarity_list:
        :return: 返回选择的原任务下标
        """
        if sourcetasks_get_method==0:
            similarity_list_sum=sum(similarity_list)
            p_list=[v/similarity_list_sum for v in similarity_list]
            if sum(p_list)!= 1:
                p_list[-1]=1-sum(p_list[:-1])
            selected_svm_index=np.random.choice(sourcetask_num,slected_k_source,replace=False,p=p_list)
        elif sourcetasks_get_method==1:
            similarity_list_temp=similarity_list[:]
            selected_svm_index = []
            Inf = 0
            for i in range(slected_k_source):
                selected_svm_index.append(similarity_list_temp.index(max(similarity_list_temp)))
                similarity_list_temp[similarity_list_temp.index(max(similarity_list_temp))] = Inf
        return selected_svm_index

    def condense_sampledata(self, selected_svm_classifier, sample_data,slected_k_source):
        condenseed_sample_data=[]
        discard_data=[]
        for one_data in sample_data.values:
            count=0
            for clff in selected_svm_classifier:
                if clff.predict([one_data])[0]==1:
                    count += 1
            if count>=slected_k_source//2:
                condenseed_sample_data.append(one_data)
            else:
                discard_data.append(one_data)
        return pd.DataFrame(condenseed_sample_data, columns=sample_data.columns),pd.DataFrame(discard_data, columns=sample_data.columns)

    def get_selected_sourcetask_svm_clf(self, selected_svm_index):
        """
        得到被选择的原任务对应的分类器
        :param selected_svm_index:被选择的原任务的索引
        :return:
        """
        selected_sourcetask_svm_classifier=list(map(self.get_one_clf,selected_svm_index))
        return selected_sourcetask_svm_classifier

    def get_one_clf(self,index):
        """
        返回其中一个分类器
        :param index: 原任务的索引
        :return:
        """
        sample_data_1 = pd.read_csv("./data/source_data/sourcetask_{}.csv".format(index))
        sample_data_2 = pd.read_csv("./output{}{}{}{}/source/sourcetask_{}/temp_data_real_perf_min.csv".format(self.target_output_name,
                                                                                                           self.select_unimportant_parameter_samples_name,
                                                                                                             self.candidate_supplement_name,
                                                                                                               self.sourcetasks_get_method_name,
                                                                                                           index))
        sample_data = pd.concat([sample_data_1, sample_data_2], axis=0)
        sample_data = sample_data.reset_index(drop=True)
        X = sample_data.iloc[:, :-1].values
        y = sample_data.iloc[:, -1].values
        y_min,y_max=min(y),max(y)
        alpha_i=self.alpha_min+(1-2*max(self.similarity_list[index-1]-0.5,0))*(self.alpha_max-self.alpha_min)
        # print(self.alpha_min,self.alpha_min,self.similarity_list[index-1])
        # print("alpha_i")
        # print(alpha_i)
        y_plus_i=y_min+(y_max-y_min)*alpha_i
        y = [1 if v < y_plus_i else 0 for v in y]
        clff = svm.SVC(gamma='scale')
        clff.fit(X, y)
        return clff

    def get_ei(self, model, eta, sample_data):
        """
        获取每个采样样本的EI值
        :param model: 当前随机森林模型
        :param eta: 之前的最优值
        :param sample_data: 采样样本
        :return: 返回EI值
        """
        pred = []
        for e in model.estimators_:
            pred.append(e.predict(sample_data.values))
        pred = np.array(pred).transpose(1, 0)
        m = np.mean(pred, axis=1)
        s = np.std(pred, axis=1)

        if np.any(s == 0.0):
            s_copy = np.copy(s)
            s[s_copy == 0.0] = 1.0
            f = self.calculate_f(eta,m,s)
            f[s_copy == 0.0] = 0.0
        else:
            f = self.calculate_f(eta,m,s)
        return f

    def calculate_f(self,eta,m,s):
        """

        :param eta:
        :param m:
        :param s:
        :return:
        """
        z = (eta - m) / s
        return (eta - m) * norm.cdf(z) + s * norm.pdf(z)

    def vis_cls(self,selected_svm_index, svm_classifier, vis_cls_path, confs, iter,condenseed_sample_data,discard_data,similarity_list):
        if not os.path.exists(os.path.join(vis_cls_path, 'iter_{}'.format(iter))):
            os.makedirs(os.path.join(vis_cls_path, 'iter_{}'.format(iter)))
        x = np.linspace(-30,30,500)
        y = np.linspace(-30,30,500)
        x,y=np.meshgrid(x, y)
        x1,y1=x.flatten(),y.flatten()
        condenseed_x,condenseed_y=condenseed_sample_data.iloc[:,0].values,condenseed_sample_data.iloc[:,1].values
        # discard_x,discard_y=discard_data.iloc[:,0].values,discard_data.iloc[:,1].values
        for i in range(len(svm_classifier)):
            z=self.task_f("sourcetask_{}".format(i+1),x,y)
            contoues = plt.contour(x, y, z, 100, cmap='RdGy')
            plt.clabel(contoues, inline=True, fontsize=8)
            plt.imshow(z, extent=[-30, 30, -30, 30], origin='lower', cmap='RdGy', alpha=0.5)
            plt.colorbar(label='contour')
            cls=svm_classifier[i]
            zz=cls.predict(list(zip(x1,y1,[0]*250000,[0]*250000,[0]*250000,[0]*250000,[0]*250000,[0]*250000,[0]*250000,[0]*250000)))
            zz[0],zz[1]=1,0
            z1=zz.reshape(500,500)
            plt.imshow(z1, extent=[-30, 30, -30, 30], origin='lower',  alpha=0.4, cmap='Blues')
            plt.colorbar(label='classifier')
            plt.scatter(condenseed_x,condenseed_y,marker='o',alpha=0.2, s=2,c="green",label="condensed-data")
            plt.legend(loc='upper left',shadow=True)
            # plt.scatter(discard_x,discard_y,marker='*',alpha=0.2,s=8,c="green")
            plt.xlabel("X0")
            plt.ylabel("X1")
            if i in selected_svm_index:
                plt.title("sourcetask{}_selected_Similarity:{ff:.2f}".format(i+1,ff=similarity_list[i]))
                plt.savefig(os.path.join(vis_cls_path, 'iter_{}/sourcetask_{}_selected.png'.format(iter,i+1)))
            else:
                plt.title("sourcetask{}_Similarity:{ff:.2f}".format(i+1,ff=similarity_list[i]))
                plt.savefig(os.path.join(vis_cls_path, 'iter_{}/sourcetask_{}.png'.format(iter,i+1)))
            plt.close()
        z = self.task_f("targettask", x, y)
        contoues = plt.contour(x, y, z, 100, cmap='RdGy')
        plt.clabel(contoues, inline=True, fontsize=8)
        plt.imshow(z, extent=[-30, 30, -30, 30], origin='lower', cmap='RdGy', alpha=0.5)
        plt.colorbar(label='contour')
        plt.scatter(condenseed_x, condenseed_y, marker='o', alpha=0.2, s=2, c="green", label="condensed-data")
        plt.legend(loc='upper left',shadow=True)
        plt.xlabel("X0")
        plt.ylabel("X1")
        # plt.scatter(discard_x,discard_y,marker='*',alpha=0.2,s=8,c="green")
        plt.title("Selected points:{num},  Compression ratio:{ff:.5f}".format(num=len(condenseed_sample_data),ff=len(condenseed_sample_data)/(len(condenseed_sample_data)+len(discard_data))))
        plt.savefig(os.path.join(vis_cls_path, 'iter_{}/targettask.png'.format(iter)))
        plt.close()


    def task_f(self,task_name, X0,X1,X2=0,X3=0,X4=0,X5=0,X6=0,X7=0,X8=0,X9=0):
        """
        任务函数
        :param task_name: 任务名称
        :param x:
        :param y:
        :return:
        """
        if task_name=="sourcetask_1":
            return -20*np.exp(0.2*np.sqrt(0.1*(X0*X0+X1*X1+X2*X2+X3*X3+X4*X4+(0.00001*X5)*(0.00001*X5)+(0.00002*X6)*(0.00002*X6)+(0.00003*X7)*(0.00003*X7)+(0.00004*X8)*(0.00004*X8)+(0.00005*X9)*(0.00005*X9))))-np.exp(0.1*(np.cos(6.28*X0)+np.cos(6.28*X1)+np.cos(6.28*X2)+np.cos(6.28*X3)+np.cos(6.28*X4)+np.cos(6.28*0.00001*X5)+np.cos(6.28*0.00002*X6)+np.cos(6.28*0.00003*X7)+np.cos(6.28*0.00004*X8)+np.cos(6.28*0.00005*X9)))+20+np.exp(1)+20
        elif task_name=="sourcetask_2":
            return -20*np.exp(0.3*np.sqrt(0.1*((X0+0.1)*(X0+0.1)+X1*X1+X2*X2+X3*X3+X4*X4+(0.00001*X5)*(0.00001*X5)+(0.00002*X6)*(0.00002*X6)+(0.00003*X7)*(0.00003*X7)+(0.00004*X8)*(0.00004*X8)+(0.00005*X9)*(0.00005*X9))))-np.exp(0.1*(np.cos(6.28*(X0+0.01))+np.cos(6.28*X1)+np.cos(6.28*X2)+np.cos(6.28*X3)+np.cos(6.28*X4)+np.cos(6.28*0.00001*X5)+np.cos(6.28*0.00002*X6)+np.cos(6.28*0.00003*X7)+np.cos(6.28*0.00004*X8)+np.cos(6.28*0.00005*X9)))+20+np.exp(1)
        elif task_name=="sourcetask_3":
            return -20*np.exp(0.6*np.sqrt(0.1*((X0+0.8)*(X0+0.8)+(X1+0.4)*(X1+0.4)+X2*X2+(X3+0.4)*(X3+0.4)+X4*X4+(0.00001*X5)*(0.00001*X5)+(0.00002*X6)*(0.00002*X6)+(0.00003*X7)*(0.00003*X7)+(0.00004*X8)*(0.00004*X8)+(0.00005*X9)*(0.00005*X9))))-np.exp(0.1*(np.cos(6.28*(X0+0.1))+np.cos(6.28*(X1+0.1))+np.cos(6.28*X2)+np.cos(6.28*(X3+0.1))+np.cos(6.28*X4)+np.cos(6.28*0.00001*X5)+np.cos(6.28*0.00002*X6)+np.cos(6.28*0.00003*X7)+np.cos(6.28*0.00004*X8)+np.cos(6.28*0.00005*X9)))+20+np.exp(1)
        elif task_name=="sourcetask_4":
            return -30*np.exp(0.2*np.sqrt(0.1*(X0*X0+X1*X1+X2*X2+X3*X3+2*X4*2*X4+(0.00001*X5)*(0.00001*X5)+(0.00002*X6)*(0.00002*X6)+(0.00003*X7)*(0.00003*X7)+(0.00004*X8)*(0.00004*X8)+(0.00005*X9)*(0.00005*X9))))-np.exp(0.1*(np.cos(6.28*X0)+np.cos(6.28*X1)+np.cos(6.28*X2)+np.cos(6.28*X3)+np.cos(6.28*2*X4)+np.cos(6.28*0.00001*X5)+np.cos(6.28*0.00002*X6)+np.cos(6.28*0.00003*X7)+np.cos(6.28*0.00004*X8)+np.cos(6.28*0.00005*X9)))+20+np.exp(1)+10
        elif task_name=="sourcetask_5":
            return -20*np.exp(5*np.sqrt(0.1*(X0*X0+0.0001*X1*0.0001*X1+X2*X2+X3*X3+X4*X4+(X5)*(X5)+(0.00002*X6)*(0.00002*X6)+(0.00003*X7)*(0.00003*X7)+(0.00004*X8)*(0.00004*X8)+(0.00005*X9)*(0.00005*X9))))-np.exp(0.1*(np.cos(6.28*X0)+np.cos(6.28*0.0001*X1)+np.cos(6.28*X2)+np.cos(6.28*X3)+np.cos(6.28*X4)+np.cos(6.28*X5)+np.cos(6.28*0.00002*X6)+np.cos(6.28*0.00003*X7)+np.cos(6.28*0.00004*X8)+np.cos(6.28*0.00005*X9)))+20+np.exp(1)+10
        elif task_name=="sourcetask_6":
            return -40*np.exp(5*np.sqrt(0.1*(X0*X0+0.0001*X1*0.0001*X1+X2*X2+X3*X3+X4*X4+(X5)*(X5)+(0.00002*X6)*(0.00002*X6)+(0.00003*X7)*(0.00003*X7)+(0.00004*X8)*(0.00004*X8)+(0.00005*X9)*(0.00005*X9))))-np.exp(0.1*(np.cos(10*X0)+np.cos(10*0.0001*X1)+np.cos(10*X2)+np.cos(10*X3)+np.cos(10*X4)+np.cos(10*X5)+np.cos(10*0.00002*X6)+np.cos(10*0.00003*X7)+np.cos(10*0.00004*X8)+np.cos(10*0.00005*X9)))+20+np.exp(1)+10
        elif task_name=="sourcetask_7":
            return -20*np.exp(0.3*np.sqrt(0.1*(0.01*X0*0.01*X0+X1*X1+X2*X2+X3*X3+X4*X4+(0.1*X5)*(0.1*X5)+(0.02*X6)*(0.02*X6)+(0.3*X7)*(0.3*X7)+(0.04*X8)*(0.4*X8)+(0.05*X9)*(0.05*X9))))-np.exp(0.1*(np.cos(6.28*0.01*X0)+np.cos(6.28*X1)+np.cos(6.28*X2)+np.cos(6.28*X3)+np.cos(6.28*X4)+np.cos(6.28*0.01*X5)+np.cos(6.28*0.02*X6)+np.cos(6.28*0.03*X7)+np.cos(6.28*0.04*X8)+np.cos(6.28*0.05*X9)))+20+np.exp(1)
        elif task_name=="sourcetask_8":
            return -20*np.exp(0.3*np.sqrt(0.1*(0.01*X0*X0+X1*X1+X2*X2+X3*X3+X4*X4+(0.1*X5)*(0.1*X5)+(0.02*X6)*(0.02*X6)+(0.3*X7)*(0.3*X7)+(0.4*X8)*(0.04*X8)+(0.05*X9)*(0.05*X9))))-np.exp(0.1*(np.cos(6.28*X0)+np.cos(6.28*X1)+np.cos(6.28*X2)+np.cos(6.28*X3)+np.cos(6.28*X4)+np.cos(6.28*0.01*X5)+np.cos(6.28*0.02*X6)+np.cos(6.28*0.03*X7)+np.cos(6.28*0.04*X8)+np.cos(6.28*0.05*X9)))+20+np.exp(1)
        elif task_name=="sourcetask_9":
            return -20*np.exp(0.3*np.sqrt(5*((X0+10)*(X0+10)+X1*X1+X2*X2+X3*X3+X4*X4+(1*X5)*(1*X5)+(2*X6)*(2*X6)+(3*X7)*(3*X7)+(4*X8)*(4*X8)+(5*X9)*(5*X9))))-np.exp(0.1*(np.cos(6.28*(X0+10))+np.cos(6.28*X1)+np.cos(6.28*X2)+np.cos(6.28*X3)+np.cos(6.28*X4)+np.cos(6.28*1*X5)+np.cos(6.28*2*X6)+np.cos(6.28*3*X7)+np.cos(6.28*4*X8)+np.cos(6.28*5*X9)))+20+np.exp(1)
        elif task_name=="sourcetask_10":
            return -20*np.exp(0.3*np.sqrt(10*(3*X0*X0+(X1+5)*(X1+5)+X2*X2+5*X3*X3+X4*X4+(1*X5)*(1*X5)+(2*X6)*(2*X6)+(3*X7)*(3*X7)+(4*X8)*(4*X8)+(5*X9)*(5*X9))))-np.exp(0.1*(np.cos(6.28*X0)+np.cos(6.28*(X1+5))+np.cos(6.28*X2)+np.cos(6.28*X3)+np.cos(6.28*X4)+np.cos(6.28*1*X5)+np.cos(6.28*2*X6)+np.cos(6.28*3*X7)+np.cos(6.28*4*X8)+np.cos(6.28*5*X9)))+20+np.exp(1)
        elif task_name=="targettask":
            return -20*np.exp(0.2*np.sqrt(0.1*(X0*X0+X1*X1+X2*X2+X3*X3+X4*X4+(0.00001*X5)*(0.00001*X5)+(0.00002*X6)*(0.00002*X6)+(0.00003*X7)*(0.00003*X7)+(0.00004*X8)*(0.00004*X8)+(0.00005*X9)*(0.00005*X9))))-np.exp(0.1*(np.cos(6.28*X0)+np.cos(6.28*X1)+np.cos(6.28*X2)+np.cos(6.28*X3)+np.cos(6.28*X4)+np.cos(6.28*0.00001*X5)+np.cos(6.28*0.00002*X6)+np.cos(6.28*0.00003*X7)+np.cos(6.28*0.00004*X8)+np.cos(6.28*0.00005*X9)))+20+np.exp(1)


    def vis_tempfile(self, task_name, temp_save_path, save_filesname):
        x = np.linspace(-30, 30, 500)
        y = np.linspace(-30, 30, 500)
        x, y = np.meshgrid(x, y)
        z = self.task_f(task_name,x, y)
        contoues = plt.contour(x, y, z, 100, cmap='RdGy')
        plt.clabel(contoues, inline=True, fontsize=8)
        plt.imshow(z, extent=[-30, 30, -30, 30], origin='lower', cmap='RdGy', alpha=0.5)
        plt.colorbar(label='contour')

        temp=pd.read_csv(temp_save_path)
        xdata,ydata=temp.iloc[:,0].values,temp.iloc[:,1].values
        plt.scatter(xdata, ydata, marker='o', alpha=0.7, s=2, c="green")
        plt.savefig(save_filesname+"temp.png")
        plt.close()

    def get_agentmodel_value(self, MODEL, top_one):
        """
        已知真实系统函数和参数配置，获取对应的真实性能值
        :param MODEL: 系统函数
        :param top_one: 配置
        :return:
        """
        # print(len(MODEL))
        target_model_temp = MODEL
        x = top_one
        j = 0
        while j < len(target_model_temp):
            if target_model_temp[j] == "X":
                k = j + 1
                while k < len(target_model_temp) and target_model_temp[k].isdigit():
                    k = k + 1
                # print(target_model_temp[j:k])
                index = target_model_temp[j:k]
                temp = target_model_temp[:j] +"(" + str(x[index])+")"
                if k < len(target_model_temp):
                    target_model_temp = temp + target_model_temp[k:]
                else:
                    target_model_temp = temp
                j = j + len(str(x[index]))+2
            else:
                j = j + 1
        # print(type(target_model_temp))
        perf = eval(target_model_temp)
        return perf

    def cal_mre(self, target_model, data):
        """
        计算mre
        :param model:
        :param condenseed_sample_data:
        :return:
        """
        # print(target_model)
        # print(data)
        # print(len(data.values))
        predict=self.model.predict(data.values)
        # print(len(predict))
        temp=[dict(data.iloc[i, :]) for i in range(len(data.values))]
        # print(len(temp))
        realy=list(map(self.get_agentmodel_value,[target_model]*len(data.values),temp))
        mre=sum(np.fabs(predict-realy))/len(predict)
        return mre



        




