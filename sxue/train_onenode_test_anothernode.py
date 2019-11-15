#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import time
import os
from sklearn.model_selection import train_test_split,cross_validate,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import metrics
from matplotlib import pyplot as plt
import predict
TIME_INTERVAL = 3600

def get_events(event_IP,events_path='merge_events.xlsx'):
    events = pd.read_excel(events_path)
    events = events[events['IP'] == event_IP] # 选取主节点
    events[u'首次发生时间'] = events[u'首次发生时间'].apply(lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))
    #events[u'首次发生时间'] = pd.to_datetime(events[u'首次发生时间'], format='%Y-%m-%d %H:%M:%S')
    events.drop_duplicates(subset=[u'首次发生时间'],keep='first',inplace=True)
    events.sort_values(by=u'首次发生时间',ascending=False,inplace=True) #保证后续取interval,line30
    print '异常事件: ',events.shape
    return events

def log_label(row,events):
    # 当前时间后续ｎ分钟内出现日志异常即label=1
    interval = events[u'首次发生时间'] - row['timestamp']
    # 时间在之后TIME_INTERVALmin内出现日志异常
    interval =  interval[interval>0]
    interval.reset_index(drop=True,inplace=True)
    if np.isnan(interval.min()):
        return 0
    elif interval.min() > TIME_INTERVAL:#30min-1800
        return 0
    else:
        return 1

def rf_construct_data(res_pathlist,event_IP):
    detected_anomalys = predict.concat_all_anomaly_csv(res_pathlist)
    detected_anomalys['timestamp'] = detected_anomalys['timestamp'].apply(lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))
    events = get_events(event_IP)
    # 当前时间后续ｎ分钟内出现日志异常，并且anomaly_score其中有一个>0.5, 即label=1
    detected_anomalys['label'] = detected_anomalys.apply(lambda row: log_label(row, events),axis=1)
    anomalys_timestamp = detected_anomalys.pop('timestamp')
    return detected_anomalys


def loadData(TrainNODE_IP_DIR,TestNODE_IP_DIR,TrainNODE_IP,TestNODE_IP):
    trainNode_pathlist = predict.get_res_pathlist(ip_dir=TrainNODE_IP_DIR)
    testNode_pathlist = predict.get_res_pathlist(ip_dir=TestNODE_IP_DIR)
    train_data = rf_construct_data(trainNode_pathlist,TrainNODE_IP)
    test_data = rf_construct_data(testNode_pathlist,TestNODE_IP)
    # 修改列名
    train_column_names = train_data.columns.tolist()
    train_column_names.remove('label')
    test_column_names = test_data.columns.tolist()
    test_column_names.remove('label')
    train_column_names.sort()
    test_column_names.sort()
    for i in range(len(train_column_names)):
        train_data.rename(columns={train_column_names[i]:'f'+str(i)},inplace=True)
    for i in range(len(test_column_names)):
        test_data.rename(columns={test_column_names[i]:'f'+str(i)},inplace=True)
    return train_data,test_data


def rf_model1(X_train,X_test): #
    Y_train = X_train.pop('label')
    Y_test = X_test.pop('label')
    # rf model
    model = RandomForestClassifier(oob_score=True, n_estimators=100, max_features='auto')
    model.fit(X_train, Y_train)
    #model = LogisticRegression()
    #model.fit(X_train,Y_train)
    #model = svm.SVC()
    #model.fit(X_train,Y_train)

    # y_train_preprobs = rf.predict_proba(Y_train)[:,1]
    y_test_pre = model.predict(X_test)
    y_test_preprobs = model.predict_proba(X_test)[:, 1]
    # auc_train = roc_auc_score(Y_train, y_train_preprobs)
    auc_test = metrics.roc_auc_score(Y_test, y_test_preprobs)
    print "test auc:\n", auc_test
    print "混淆矩阵:\n", metrics.confusion_matrix(Y_test, y_test_pre, labels=[0, 1])
    print "综合报告:\n", metrics.classification_report(Y_test, y_test_pre)

def main():
    # load train and test dataset
    TrainNODE_IP_DIR = 'es_nodes3_52'
    TestNODE_IP_DIR = 'es_nodes3_36'
    TrainNODE_IP = '10.33.208.52'
    TestNODE_IP = '10.33.208.36'
    train_data, test_data = loadData(TrainNODE_IP_DIR,TestNODE_IP_DIR,TrainNODE_IP,TestNODE_IP)
    rf_model1(train_data,test_data)

if __name__ == '__main__':
    main()