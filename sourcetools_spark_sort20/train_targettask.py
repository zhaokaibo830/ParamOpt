from random_forest import *
# 第一阶段，训练模型，保存在output的文件夹下

if __name__ == '__main__':

    # 训练
    confs = json.load(open('paras.json'), strict=False)
    model = random_forest("sourcetask_sort_20",confs["targettask"])
    model.target_train()