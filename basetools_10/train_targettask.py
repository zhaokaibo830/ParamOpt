from random_forest import *
# 第一阶段，训练模型，保存在output的文件夹下
import numpy as np
import random
from tqdm import tqdm


if __name__ == '__main__':
    confs = json.load(open('paras.json'), strict=False)
    # 原任务训练
    print("原任务开始训练：")
    for task_name, target_model in tqdm(confs["sourcetasks"].items()):
        # print(task_name,target_model)
        model = random_forest("source", task_name, target_model)
        model.source_train()

    # 目标任务训练
    print("目标任务开始训练：")
    model = random_forest("target","targettask",confs["targettask"])
    model.target_train()
