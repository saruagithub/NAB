#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
#from keras.models import Sequential
#from keras.layers import Dense,Dropout
#from keras.wrappers.scikit_learn import KerasClassifier
import predict,evaluate
import time
from sklearn.model_selection import train_test_split,cross_validate,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import metrics

def construct_data():
    data_csv_path = '../data/realBKmonitor/1122bkmonitor_selfscript_es_nodes_3.csv'
    nodes_host_ip = '10.33.208.36'
    single_data_path = '../data/es_nodes3_36/'
    cols = ['timestamp',
            'nodes_fs_total_available_in_b',
           'nodes_jvm_mem_heap_used_per',
            'nodes_os_cpu_load_average_1m',
            'nodes_os_cpu_percent',
            'nodes_process_cpu_percent']
    data = pd.read_csv(data_csv_path)
    data.rename(columns={'time': 'timestamp'}, inplace=True)
    data['timestamp'] = data['timestamp'].apply(lambda x:x.rstrip('Z'))
    data['timestamp'] = data['timestamp'].apply(lambda x:x.replace('T',' '))
    #data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y-%m-%dT%H:%M:%SZ')
    data_ip = data[data['nodes_host'] == nodes_host_ip]
    data_col = data_ip[cols]
    data_col.fillna(method='ffill',inplace=True)

    # construct label
    events = evaluate.get_events('merge_events.xlsx')
    data_col['timestamp'] = data_col['timestamp'].apply(lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))
    data_col['label'] = data_col.apply(lambda row: predict.log_label(row, events), axis=1)
    data_col.drop(['timestamp'],axis=1,inplace=True)
    return data_col

def rf_model(anomalys):
    anomalys_label = anomalys.pop('label')
    auc_list = []
    for k in range(5):
        X_train, X_test, Y_train, Y_test = train_test_split(anomalys, anomalys_label, test_size=0.3)
        print 'Xtrain,Xtest,Ytrain,Ytest: ',X_train.shape,X_test.shape,Y_train.shape,Y_test.shape
        # rf model
        #model = RandomForestClassifier(oob_score=True, n_estimators=100, max_features=3)
        #model.fit(X_train, Y_train)
        #model = LogisticRegression()
        #model.fit(X_train,Y_train)
        #model = svm.SVC()
        #model.fit(X_train,Y_train)
        model = GaussianNB()
        model.fit(X_train,Y_train)

        # y_train_preprobs = rf.predict_proba(Y_train)[:,1]
        y_test_pre = model.predict(X_test)
        #y_test_preprobs = model.predict_proba(X_test)[:, 1]
        # auc_train = roc_auc_score(Y_train, y_train_preprobs)
        #auc_test = metrics.roc_auc_score(Y_test, y_test_preprobs)
        #auc_list.append(auc_test)

        # print "y test prediction:\n", y_test_pre
        #print "test auc:\n", auc_test
        print "混淆矩阵:\n", metrics.confusion_matrix(Y_test, y_test_pre, labels=[0, 1])
        print "综合报告:\n", metrics.classification_report(Y_test, y_test_pre)
    print 'auc_avg:',np.mean(auc_list)

def main():
    data = construct_data()
    rf_model(data)


if __name__ == '__main__':
    main()