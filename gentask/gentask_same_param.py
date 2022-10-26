import os
import json
import random
import pandas as pd
import numpy as np


# 随机采样
def random_sample(selected_params, sample_num,all_tasks):
    """
    对不太重要的参数进行随机采样
    :param selected_params: selected_params字典形式
    :param sample_num: 采样数量
    :return: 采样结果
    """

    # 随机采样开始
    samples = []
    for _ in range(sample_num):
        samples.append([])
    for key in selected_params.keys():
        for i in range(sample_num):
            samples[i].append(random.uniform(selected_params[key][1][0] + 0.1, selected_params[key][1][1]))
    for i in range(sample_num):
        samples[i].append(0)
    for task_name,target_model in all_tasks.items():
        for i in range(len(samples)):
            target_model_temp=target_model
            x = samples[i][:-1]
            j=0
            while j<len(target_model_temp):
                if target_model_temp[j]=="X":
                    k=j+1
                    while k<len(target_model_temp) and target_model_temp[k].isdigit():
                        k=k+1
                    print(target_model_temp[j+1:k])
                    index=int(target_model_temp[j+1:k])
                    temp=target_model_temp[:j]+str(x[index])
                    if k<len(target_model_temp):
                        target_model_temp=temp+target_model_temp[k:]
                    else:
                        target_model_temp=temp
                    print(target_model_temp)
                    j=j+len(str(x[index]))
                else:
                    j=j+1
            y = eval(target_model_temp)
            samples[i][-1]=y
        df_samples = pd.DataFrame(samples, columns=list(selected_params.keys())+["PERF"])
        temp_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'taskdata/{}.csv'.format(task_name))
        df_samples.to_csv(temp_filename, index=False)

if __name__=="__main__":
    confs = json.load(open('paras.json'), strict=False)
    selected_params = confs['common_params']['all_params']
    all_tasks = confs['alltasks']
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)),'taskdata')):
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)),'taskdata'))



    df_samples = random_sample(selected_params, 100, all_tasks)




