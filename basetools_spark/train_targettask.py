from random_forest import *
# 第一阶段，训练模型，保存在output的文件夹下
import numpy as np
import random
from tqdm import tqdm


if __name__ == '__main__':

    # 目标任务训练
    print("目标任务开始训练：")
    model = random_forest("sourcetask_bayes_0dot1")
    model.target_train()
