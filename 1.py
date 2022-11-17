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



def get_performance(params):
    """
    接系统获取性能值
    :param params: {'param1': 10, 'param2': 's1', ...}
    :return: 性能值
    """

    headers = {"user-agent": "Mizilla/5.0"}
    value = parse.urlencode(params)
    print(value)
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
            print(111111)
            f = request.urlopen(req1, timeout=360)
            print(2222222)
            # print(f)
            Data = f.read()
            print(Data)
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

param={"executorCores":"1",
       "executorMemory":"1024",
       "executorInstances":"2",
       "defaultParallelism":"8",
       "memoryOffHeapEnabled":"false",
       "memoryOffHeapSize":"0",
       "memoryFraction":"0.6",
       "memoryStorageFraction":"0.5",
       "shuffleFileBuffer":"32",
       "speculation":"false",
       "reducerMaxSizeInFlight":"48",
       "shuffleSortBypassMerageThreshold":"200",
       "speculationInterval": "100",
       "speculationMultiplier": "1.5",
       "speculationQuantile": "0.75",
       "broadcastBlockSize": "4",
       "ioCompressionCodec": "lz4",
       "ioCompressionLz4BlockSize": "64",
       "ioCompressionSnappyBlockSize": "32",
       "kryoRederenceTracking": "true",
       "kryoserializerBufferMax": "64",
       "kryoserializerBuffer": "64",
       "storageMemoryMapThreshold": "2",
       "networkTimeout": "120",
       "localityWait": "3",
       "shuffleCompress": "true",
       "shuffleSpillCompress": "true",
       "broadcastCompress": "true",
       "rddCompress": "false",
       "serializer": "org.apache.spark.serializer.JavaSerializer"
       }

cc=get_performance(param)
print(cc)