import os
import json
from sklearn import svm
import pandas as pd
import pickle
if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'classifier/svm/')):
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'classifier/svm/'))

confs=json.load(open('paras.json'), strict=False)
for i in range(1,confs["sourcetask_num"]+1):
    sample_data_1=pd.read_csv("./data/source_data/sourcetask_{}.csv".format(i))
    sample_data_2=pd.read_csv("./source_output/sourcetask_{}/temp.csv".format(i))
    sample_data = pd.concat([sample_data_1, sample_data_2], axis=0)
    sample_data=sample_data.reset_index(drop=True)
    X=sample_data.iloc[:,:-1].values
    y=sample_data.iloc[:,-1].values
    y=[1 if v<0 else 0 for v in y]
    clff=svm.SVC(gamma='scale')
    clff.fit(X,y)
    with open("./classifier/svm/sourcetask_{}.pickle".format(i), 'wb') as f:
        pickle.dump(clff,f)


