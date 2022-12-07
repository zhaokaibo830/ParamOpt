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
    def __init__(self,task_name):

        # 加载相关配置参数

        self.confs = json.load(open('paras.json'), strict=False)
        self.arg = json.load(open('paras.json'), strict=False)

        # 参数名及其范围以及默认值
        self.selected_params = self.confs['common_params']['all_params']

        # 任务名称
        self.task_name = task_name



        # 候选补齐方式，0表示不补齐，1表示样本太少在投票一直的范围内随机采样,2表示样本太少在投票一直的范围选择压缩空间内任意两点之间的随机一个值,3表示样本太少在以压缩空间为中心的周围盒子内部随即采样
        self.candidate_supplement=self.confs['candidate_supplement']

        # 候选补齐门限
        self.candidate_supplement_threshold = self.confs['candidate_completion_threshold']

        # 源任务选择方式，0表示按照相似度计算得到的概率获取，概率越大被选择的概率越大，1表示按照相似度大小取，取相似度最大的几个
        self.sourcetasks_get_method=self.confs['sourcetasks_get_method']

        # 选择原任务的个数
        self.slected_k_source=self.confs['slected_k_source']

        # 原任务数量
        self.sourcetask_num=self.confs['sourcetask_num']

        # 原任务模型
        self.sourcetask_models = [None] * self.sourcetask_num

        # 原任务分类器需要的参数
        self.alpha_max=self.confs['alpha_max']
        self.alpha_min=self.confs['alpha_min']


        # 当前所有样本在源任务模型上计算得到的值
        self.source_M_list=[]

        # 相似度列表
        self.similarity_list=[]

        # 性能列名
        self.performance = self.confs['common_params']['performance'].split('_')[0]

        # 性能值是极大还是极小
        self.min_or_max = self.confs['common_params']['performance'].split('_')[1]

        # 模型
        self.model = None

        if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)),'target_output/target')):
            os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)),'target_output/target/'))
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'target_output/target/random_forest.pickle')


        # 特征重要性保存路径
        self.import_feature_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                'target_output/target/random_forest_feature_importance.txt')

        # 初始数据集路径
        self.filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'data/target_data/{}.csv'.format(self.task_name))

        # 数据保存位置
        self.save_filesname = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          'target_output/target/')


        # 数据列名
        self.columns = None
        self.fake_columns = None

        # 标准化模型
        self.scalar = None

        self.data_type = {"executorCores": int,
                     "executorMemory": int,
                     "executorInstances": int,
                     "defaultParallelism": int,
                     "memoryOffHeapEnabled": str,
                     "memoryOffHeapSize": int,
                     "memoryFraction": float,
                     "memoryStorageFraction": float,
                     "shuffleFileBuffer": int,
                     "speculation": str,
                     "reducerMaxSizeInFlight": int,
                     "shuffleSortBypassMerageThreshold": int,
                     "speculationInterval": int,
                     "speculationMultiplier": float,
                     "speculationQuantile": float,
                     "broadcastBlockSize": int,
                     "ioCompressionCodec": str,
                     "ioCompressionLz4BlockSize": int,
                     "ioCompressionSnappyBlockSize": int,
                     "kryoRederenceTracking": str,
                     "kryoserializerBufferMax": int,
                     "kryoserializerBuffer": int,
                     "storageMemoryMapThreshold": int,
                     "networkTimeout": int,
                     "localityWait": int,
                     "shuffleCompress": str,
                     "shuffleSpillCompress": str,
                     "broadcastCompress": str,
                     "rddCompress": str,
                     "serializer": str}


        # 加载代理模型，不接系统的话
        # self.MODEL =target_model
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
        columns = self.columns
        # 首先处理没有优先关系的enum参数，即string参数
        char = []  # enum的列名称
        enum_index = []  # enum的原始列索引
        for name in columns:
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
            # print(data.values.shape)
            # enum_book[c] = list(pd.get_dummies(data[c]).columns)
            # print(data[c])
            enum_data = pd.get_dummies(data[c], prefix=c)  # 独热编码后的变量
            enum_book[c] = list(enum_data.columns)
            # print(enum_data.columns)
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

        # print(enum_book)
        self.fake_columns = list(data.columns)

        return data


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
        # data_x = data.iloc[:, :-1].values
        # self.fake_columns = list(data.iloc[:, :-1].columns)
        data_x = self.data_in(data.iloc[:,:-1])
        # print("data_X")
        # print(data_x)

        # print("data_y")
        # print(data_y)
        # 标准化
        self.scalar = StandardScaler()
        data_x = self.scalar.fit_transform(data_x.astype(float))

        # 随机森林训练
        self.model = RandomForestRegressor()
        self.model.fit(data_x, data_y)

        # 保存训练好的随机森林模型
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)

    def target_train(self):

        origin_data = pd.read_csv(self.filename, dtype=self.data_type)
        self.arg["start_sample_num"]=len(origin_data.values)
        with open(self.save_filesname+"arg.json","w") as f:
            json.dump(self.arg,f,indent=2)

        unimportant_feature_sample_nums = self.confs['sample_params']['unimportant_feature_sample_nums']  # 初始不重要参数采样数量
        important_feature_sample_num = self.confs['sample_params']['important_feature_sample_num']  # 初始重要参数采样数量
        sample_step_change = self.confs['sample_params']['sample_step_change']  # 每步重要参数采样增加个数
        temp_data_real_perf_min = pd.DataFrame(columns=origin_data.columns)  # 真实中间结果保存结果集
        eta = max(origin_data.iloc[:, -1].values) if self.min_or_max == 'max'else min(origin_data.iloc[:, -1].values) # eta是当前最优值
        #  eta_index表示最小值获取的代数或者初始样本的位置，如果小于表示最小值是在初始样本得到的
        eta_index=-np.argmax(origin_data.iloc[:, -1].values) if self.min_or_max == 'max'else -np.argmin(origin_data.iloc[:, -1].values)

        every_epoch_remain_sample_num = []  # 每一轮剩余的样本数
        every_epoch_remain_sample_ratio=[]  # 每一轮剩余的样本数所占比例

        self.sourcetask_name_list=os.listdir("./target_output/source/")
        similarity_save = pd.DataFrame(columns=self.sourcetask_name_list)  #相似度保存位置

        selectd_important_features_save = []  # 保存每轮选择的重要参数

        condenseed_sample_data_iszero_iter=[]

        writer = SummaryWriter(self.save_filesname+"runs/")

        # 获取源任务模型
        self.get_sourcetask_models(self.sourcetask_name_list)

        # 获取初始样本在源任务模型上计算得到的值
        self.source_M_list=self.cal_M_list(self.sourcetask_models,(self.data_in(origin_data.iloc[:,:-1])).values)

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
            self.similarity_list=self.cal_Similarity(self.source_M_list,origin_data.iloc[:, -1].values)

            # print(self.similarity_list)



            # 保存相似度
            similarity_one_iter = pd.DataFrame(data=[self.similarity_list], columns=similarity_save.columns)
            similarity_save=pd.concat([similarity_save,similarity_one_iter],axis=0)


            # 选择原任务,得到原任务对应的索引
            self.selected_svm_index=self.choose_source(self.slected_k_source,self.similarity_list,self.sourcetask_num,self.sourcetasks_get_method)

            # 获取原任务分类器
            self.selected_svm_classifier=self.get_selected_sourcetask_svm_clf(self.selected_svm_index)
            # self.selected_svm_classifier = np.array(self.svm_classifier)[self.selected_svm_index]

            # 分别采样重要参数、不重要的参数
            # 不重要的参数随机采样,
            # 1表示不重要参数的采样为对所有参数进行随机采样，
            sample_data_1 = random_sample_1(self.selected_params,unimportant_feature_sample_nums)

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

            # 样本补齐
            if self.candidate_supplement==0:
                if len(condenseed_sample_data.values) == 0:
                    condenseed_sample_data = sample_data.iloc[:, :]
            elif self.candidate_supplement==1:
                m=0
                while len(condenseed_sample_data.values)<self.candidate_supplement_threshold and m<50:
                    candidate_sample_data=random_sample_1(self.selected_params,important_feature_sample_num)
                    candidate_sample_data,_=self.condense_sampledata(self.selected_svm_classifier,candidate_sample_data,self.slected_k_source)
                    condenseed_sample_data=pd.concat([condenseed_sample_data, candidate_sample_data], axis=0)
                    condenseed_sample_data = condenseed_sample_data.reset_index(drop=True)
                    m += 1
                if len(condenseed_sample_data.values)==0:
                    condenseed_sample_data_iszero_iter.append(i)
                    condenseed_sample_data = sample_data.iloc[:, :]
            elif self.candidate_supplement==2:
                if len(condenseed_sample_data.values) <= 1:
                    condenseed_sample_data = sample_data.iloc[:, :]
                candidate_data_temp=list(condenseed_sample_data.values)
                while len(candidate_data_temp)<self.candidate_supplement_threshold:
                    point_1_index,point_2_index=np.random.choice(len(candidate_data_temp),2,replace=False)
                    point_1,point_2=np.array(candidate_data_temp[point_1_index]),np.array(candidate_data_temp[point_2_index])
                    in_point=point_1+random.uniform(0,1)*(point_2-point_1)
                    candidate_data_temp.append(in_point)
                condenseed_sample_data=pd.DataFrame(candidate_data_temp,columns=list(self.selected_params.keys()))
            elif self.candidate_supplement == 3:
                if len(condenseed_sample_data.values) <= 1:
                    condenseed_sample_data = sample_data.iloc[:, :]
                candidate_data_temp = list(condenseed_sample_data.values)
                while len(candidate_data_temp) < self.candidate_supplement_threshold:
                    print("______________")
                    print(len(candidate_data_temp))
                    min_distance=self.get_two_point_min_distance(candidate_data_temp,self.selected_params,sample_data.columns)
                    sample_list=[]
                    for k in range(len(candidate_data_temp)):
                        sample_temp=self.box_of_point_sample(candidate_data_temp[k],min_distance,sample_data.columns)
                        condese_sample_k,_=self.condense_sampledata(self.selected_svm_classifier, sample_temp, self.slected_k_source)
                        sample_list.append(condese_sample_k)
                    for condese_sample_i in sample_list:
                        candidate_data_temp.extend(condese_sample_i.values)
                condenseed_sample_data = pd.DataFrame(candidate_data_temp, columns=list(self.selected_params.keys()))



            # 使用训练好的模型进行预测，获取每个样本对应的EI值
            ei = self.get_ei(self.model, eta, condenseed_sample_data)
            max_index = np.argmax(ei)

            # 找到性能值最优的那一个，接入到实际的系统跑一下
            top_one = dict(condenseed_sample_data.iloc[max_index, :])
            # print(top_one)
            perf = get_performance(top_one)

            # 没有实际系统，接代理模型得到性能值
            # perf = self.get_agentmodel_value(self.MODEL, top_one)

            # 更新当前所有样本在源任务模型上计算得到的值
            self.updata_M_list(self.source_M_list,self.sourcetask_models,self.data_in(condenseed_sample_data.iloc[max_index:max_index+1, :]).values)
            # print(len(self.source_M_list),len(self.source_M_list[0]))

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

            # 重要参数采样数量递增
            important_feature_sample_num = important_feature_sample_num + sample_step_change

            # 随机采样数量衰减
            # 动态根据公式调整不重要参数的数量
            theta = -50 / math.log(0.5)
            unimportant_feature_sample_nums = int(unimportant_feature_sample_nums * math.exp(
                -math.pow(max(0, i - 20), 2) / (2 * theta * theta)))

            writer.add_scalar('perf_min_forest', self.model.predict(self.data_in(condenseed_sample_data.iloc[max_index:max_index+1, :]).values)[0], global_step=i)
            writer.add_scalar('perf_min', perf, global_step=i)
            np.savetxt(self.save_filesname + 'every_epoch_remain_sample_ratio.txt', every_epoch_remain_sample_ratio)

            similarity_save.to_csv(self.save_filesname+"similarity.csv", index=False)
            np.savetxt(self.save_filesname+'every_epoch_remain_sample_num.txt', every_epoch_remain_sample_num)
            np.savetxt(self.save_filesname+'every_epoch_remain_sample_ratio.txt', every_epoch_remain_sample_ratio)
            temp_data_real_perf_min.to_csv(self.save_filesname + "temp_data_real_perf_min.csv", index=False)
        # np.savetxt(self.save_filesname + 'mre_sum.txt', mre_sum)
        np.savetxt(self.save_filesname + 'condenseed_sample_data_iszero_iter.txt', condenseed_sample_data_iszero_iter)
        with open(self.save_filesname + 'optimal_value.json', "w") as f:
            json.dump({"index":str(eta_index),"opt_perf":float(eta)}, f)
        np.savetxt(self.save_filesname + 'selectd_important_features_save.txt',np.array(selectd_important_features_save), fmt="%s")
        # self.vis_tempfile(self.task_name,self.save_filesname+"temp_data_real_perf_min.csv",self.save_filesname)

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
        data[self.performance] = 0
        data, _, _ = self.data_in(data)
        data = self.scalar.transform(data)
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
        sample_data_temp=self.data_in(sample_data)
        for i,one_data in enumerate(sample_data_temp.values):
            count=0
            for clff in selected_svm_classifier:
                if clff.predict([one_data])[0]==1:
                    count += 1
            if count>=slected_k_source//2:
                condenseed_sample_data.append(sample_data.iloc[i,:].values)
            else:
                discard_data.append(sample_data.iloc[i,:].values)
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
        sample_data_1 = pd.read_csv("./data/source_data/{}.csv".format(self.sourcetask_name_list[index]),dtype=self.data_type)
        sample_data_2 = pd.read_csv("./target_output/source/{}/temp_data_real_perf_min.csv".format(self.sourcetask_name_list[index]),dtype=self.data_type)
        sample_data = pd.concat([sample_data_1, sample_data_2], axis=0)
        sample_data = sample_data.reset_index(drop=True)
        X = (self.data_in(sample_data.iloc[:, :-1])).values
        y = sample_data.iloc[:, -1].values
        y_min,y_max=min(y),max(y)
        alpha_i=self.alpha_min+(1-2*max(self.similarity_list[index]-0.5,0))*(self.alpha_max-self.alpha_min)
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
        data = self.data_in(sample_data)
        self.scalar=StandardScaler()
        data = self.scalar.fit_transform(data.astype(float))

        pred = []
        for e in model.estimators_:
            pred.append(e.predict(data))
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


    def get_sourcetask_models(self, sourcetask_name_list):
        """

        :param sourcetask_models_path:
        :return:
        """
        for i,sourcetask_name in enumerate(sourcetask_name_list):
            # print(sourcetask_name)
            with open("./target_output/source/{}/random_forest.pickle".format(sourcetask_name), 'rb') as f:
                self.sourcetask_models[i] = pickle.load(f)

    def cal_M_list(self, sourcetask_models, target_X):
        return list(map(self.cal_one_M,sourcetask_models,[target_X]*len(self.sourcetask_models)))

    def cal_one_M(self,sourcetask_model,target_X):
        return list(sourcetask_model.predict(target_X))

    def cal_Similarity(self, source_M_list, target_Y):
        """
        计算每个原任务和目标任务之间的相似度
        :param sourcetask_models: 原模型列表
        :param target_X:
        :param target_Y:
        :return:
        """
        similarity_list = list(map(self.cal_one_Similarity, source_M_list, [target_Y] * len(self.sourcetask_models)))

        return similarity_list

    def cal_one_Similarity(self, one_M, target_Y):
        # print(source_M)
        one_M_col = np.array(one_M).reshape(len(target_Y), 1)
        one_M_row = one_M_col.T
        target_Y_col = np.array(target_Y).reshape(len(target_Y), 1)
        target_Y_row = target_Y_col.T
        temp = np.logical_not(np.logical_xor(one_M_col < one_M_row, target_Y_col < target_Y_row))
        rank_loss = np.sum(temp[np.triu_indices(len(target_Y), 1)])
        return 2 * rank_loss / (len(target_Y) * (len(target_Y) - 1))

    def updata_M_list(self, source_M_list, sourcetask_models, values):
        for i in range(len(sourcetask_models)):
            source_M_list[i].append(sourcetask_models[i].predict(values)[0])

    def get_two_point_min_distance(self, candidate_data_temp,selected_params,sample_data_columns):
        zuhe=np.triu_indices(len(candidate_data_temp), 1)
        min_distance_list=float("inf")
        for j in range(len(list(zuhe[0]))):
            point_1=candidate_data_temp[zuhe[0][j]]
            point_2=candidate_data_temp[zuhe[1][j]]
            point_1_temp,point_2_temp=[],[]
            for k in range(len(point_1)):
                key=sample_data_columns[k]
                if selected_params[key]=='int' or selected_params[key]=='float':
                    a,b=selected_params[key][1][0],selected_params[key][1][1]
                    point_1_temp.append((point_1[k]-a)/(b-a))
                    point_2_temp.append((point_2[k] - a) / (b - a))
            min_distance_list=min(min_distance_list,np.linalg.norm(np.array(point_1_temp)-np.array(point_2_temp)))
        return min_distance_list


    def box_of_point_sample(self,center,min_distance,sample_data_columns):
        config=json.load(open('paras.json'), strict=False)
        all_params=config['common_params']['all_params']
        for j in range(len(center)):
            key=sample_data_columns[j]
            if all_params[key][0]=="int" or all_params[key][0]=="float":
                a,b=all_params[key][1][0],all_params[key][1][1]
                distance=min_distance*(b-a)
            if all_params[key][0]=="int":
                if distance < 1:
                    left=a if center[j]-1<a else center[j]-1
                    right=b if center[j]+1>b else center[j]+1
                else:
                    left=a if center[j]-distance<a else center[j]-distance
                    right=b if center[j]+distance>b else center[j]+distance

                all_params[key][1][0], all_params[key][1][1] = int(left), int(right)

            elif  all_params[key][0]=="float":
                left = a if center[j] - distance < a else center[j] - distance
                right = b if center[j] + distance > b else center[j] + distance

                all_params[key][1][0], all_params[key][1][1] = left,right
        return random_sample_1(all_params,10)


