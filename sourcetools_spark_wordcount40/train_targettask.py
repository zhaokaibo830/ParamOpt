from random_forest import *
# 第一阶段，训练模型，保存在output的文件夹下

if __name__ == '__main__':

    # 训练
    model = random_forest("sourcetask_wordcount_40")
    model.target_train()