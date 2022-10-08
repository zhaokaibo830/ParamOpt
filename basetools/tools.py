import numpy as np
np.random.seed(10)

def cal_Similarity(sourcetask_models,target_X,target_Y):
    similarity_list=[]
    for sourcetask_model in sourcetask_models:
        rank_loss = 0
        for j in range(len(target_X)):
            for k in range(j+1,len(target_X)):
                target_X_j = sourcetask_model.predict([target_X[j]])
                target_X_k = sourcetask_model.predict([target_X[k]])
                if (target_X_j[0]<target_X_k[0]) ^ (target_Y[j]<target_Y[k]):
                    rank_loss += 1
        similarity_list.append(2*rank_loss/(len(target_Y)/(len(target_Y)-1)))
    return similarity_list








