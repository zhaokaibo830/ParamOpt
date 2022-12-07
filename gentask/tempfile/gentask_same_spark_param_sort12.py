import os
import json
import random
import pandas as pd
import numpy as np
from urllib import request, parse
import time
from tqdm import *

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
    req1 = request.Request('http://192.168.0.168:9002/api/spark/runSortForTime?%s' % value, headers=headers)  # 这样就能把参数带过去了
    # print('http://192.168.0.168:9002/api/spark/runWordcountForTime?%s' % value)
    # 下面是获得响应
    i = 0
    # url_ = 'http://47.104.101.50:8081/jmeter'
    while True:
        try:
            # print(data_)
            start=time.time()
            start_str=time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(start))
            print("第{}次请求开始时间：".format(i+1),start_str)

            f = request.urlopen(req1, timeout=86400)

            end = time.time()
            end_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end))


            print("第{}次请求结束时间：".format(i + 1), end_str)
            print("第{}次请求时长：{:.2f}s".format(i+1,end-start))

            # print(f)
            Data = f.read()
            data = Data.decode('utf-8')
            if data == "error":
                print("查不到，报errror，重新尝试\n")
                f.close()
                if i < 100:
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
            print("请求异常:",e)
            i = i + 1
            if i < 100:
                time.sleep(3)
            else:
                return -1.0

    return (float)(data)

# 随机采样
def get_sample(path, sample_num):
    """
    对不太重要的参数进行随机采样
    :param selected_params: selected_params字典形式
    :param sample_num: 采样数量
    :return: 采样结果
    """
    data_type = {"executorCores": int,
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

    df_samples=pd.read_csv(path,dtype=data_type)
    print(df_samples.iloc[0,:])
    for i in trange(len(df_samples.values)):
        top_one = dict(df_samples.iloc[i, :-1])
        value=get_performance(top_one)
        df_samples.iloc[i,-1]=value
    temp_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'taskdata/sourcetask_sort_12.csv')
    df_samples.to_csv(temp_filename, index=False)

if __name__=="__main__":

    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)),'taskdata')):
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)),'taskdata'))
    # print(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)),'spark_data'),"sourcetask_Workcount_32.csv"))

    get_sample(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)),'spark_data'),"sourcetask_sort_10.csv"), 50)




