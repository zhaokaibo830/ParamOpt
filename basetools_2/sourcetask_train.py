
from random_forest import *
# 第一阶段，训练模型，保存在output的文件夹下

if __name__ == '__main__':

    # 训练
    confs = json.load(open('paras.json'), strict=False)
    for task_name,target_model in confs["sourcetasks"].items():
        model = random_forest("source",task_name,target_model)
        model.source_train()




