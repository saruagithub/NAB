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
import evaluate

def get_res_paths(path):
    pathlist = []
    for item in os.listdir(path):
        pathlist.append(path+item)
    return pathlist

def get_res_pathlist(ip_dir,dir='../results/myres/'):
    '''
    :param dir: '../results/myres/'
    :return: 返回es_nodes3_52 ip_dir目录里的所有检测结果csv文件
    '''
    pathlist = []
    for detec_dir in os.listdir(dir):
        for data_path in os.listdir(dir + detec_dir):
            if data_path == ip_dir:
                for res in os.listdir(dir + detec_dir + '/' + ip_dir  +'/'):
                    pathlist.append(dir + detec_dir + '/' + ip_dir  +'/' + res)
    return pathlist

def concat_all_anomaly_csv(pathlist):
    '''
    :param path: 将path下的所有异常csv结合到一起,timestamp保留一个
    :return: timestamp,各序列异常分数的csv
    '''
    # 经过HTM预测得到的所有异常csv，保留timestamp,anomaly_score并左右拼接起来
    res_datas = []
    for path in pathlist:
        data = pd.read_csv(path)
        if 'numenta' in path:
            data.drop(['value','raw_score','label'],axis=1 ,inplace=True)
        else:
            data.drop(['value','label'],axis=1 ,inplace=True)
        data = data.rename(columns = {'anomaly_score':'anomalyscore_'+path.split('/')[5].rstrip('.csv')})
        data = data.set_index(['timestamp'])
        res_datas.append(data)

    all_anomalys = pd.concat(res_datas,axis=1)
    all_anomalys.fillna(method='ffill',inplace=True)
    all_anomalys.reset_index(inplace=True)
    all_anomalys.rename(columns = {'index':'timestamp'},inplace=True)
    # drop duplicated timestamp
    #all_anomalys = all_anomalys.loc[:,~all_anomalys.columns.duplicated()]
    return all_anomalys


def log_label(row,events):
    '''
    :param row:每一行
    :param events:日志异常excel表
    :return:如果在当前时间的后n分钟内出现日志异常即label=1
    '''
    # 当前时间后续ｎ分钟内出现日志异常即label=1
    interval = events[u'首次发生时间'] - row['timestamp']
    # 时间在之后30min内出现日志异常
    interval =  interval[interval>0]
    interval.reset_index(drop=True,inplace=True)
    # row_notime = row.drop('timestamp')
    # anomaly_score > 0.5 && log anomaly time is later than anomaly_score time in 30min
    #if row_notime[row_notime>0.5].shape[0] < 1:
    #    return 0
    if np.isnan(interval.min()):
        #print 'interval.min: ',interval.min()
        return 0
    elif interval.min() > evaluate.TIME_INTERVAL:#30min-1800
        return 0
    else:
        return 1
    # interval = np.abs(events[u'首次发生时间'] - row['timestamp'])
    # index = np.argmin(interval)
    # if interval[index] < 600:
    #     return 1
    # else:
    #     return 0


def rf_construct_data(res_pathlist,events_path='merge_events.xlsx'):
    '''
    :param detected_path: myres/下面的经过异常检测的文件夹路径
    :param events_path: 日志异常的excel路径
    :return: 构建带label的整个数据集
    '''
    detected_anomalys = concat_all_anomaly_csv(res_pathlist)
    detected_anomalys['timestamp'] = detected_anomalys['timestamp'].apply(lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))
    events = evaluate.get_events(events_path)
    print '异常事件: ',events.shape
    # 当前时间后续ｎ分钟内出现日志异常，并且anomaly_score其中有一个>0.5, 即label=1
    detected_anomalys['label'] = detected_anomalys.apply(lambda row: log_label(row, events),axis=1)
    '''
    # 构建与events一一对应的label=1
    events.rename(columns={u'首次发生时间':'timestamp',u'优先级':'label'},inplace=True)
    print events.columns
    events[u'timestamp'] = events[u'timestamp'].apply(lambda x:x.replace(second=0))
    events[u'label'] = 1
    events = events[['timestamp','label']]
    print events.shape
    detected_anomalys['timestamp'] = pd.to_datetime(detected_anomalys['timestamp'],unit='s')
    detected_anomalys = pd.merge(detected_anomalys,events,on='timestamp',how='left')
    detected_anomalys['label'].fillna(0,inplace=True)
    detected_anomalys['label'] = detected_anomalys['label'].astype(np.int)
    '''
    return detected_anomalys


def load_dataset():
    #res_pathlist = get_res_paths('../results/myres/numenta/es_nodes3_9ips/')
    res_pathlist = get_res_pathlist(ip_dir=evaluate.IP_DIR)
    anomalys = rf_construct_data(res_pathlist)
    #print 'anomalys columns name: ',anomalys.columns
    anomalys['timestamp'] = pd.to_datetime(anomalys['timestamp'],unit='s')
    anomalys['timestamp'] = anomalys['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')

    print 'anomalys shape and 0/1\n',anomalys.shape[0],anomalys['label'].value_counts()
    anomalys_timestamp = anomalys.pop('timestamp')
    #anomalys_label = anomalys.pop('label')
    return anomalys

def rf_model(anomalys):
    anomalys_label = anomalys.pop('label')
    auc_list = []
    for k in range(5):
        X_train, X_test, Y_train, Y_test = train_test_split(anomalys, anomalys_label, test_size=0.3)
        print 'Xtrain,Xtest,Ytrain,Ytest: ',X_train.shape,X_test.shape,Y_train.shape,Y_test.shape
        # rf model
        model = RandomForestClassifier(oob_score=True, n_estimators=100, max_features=6)
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
        auc_list.append(auc_test)

        # print "y test prediction:\n", y_test_pre
        print "test auc:\n", auc_test
        print "混淆矩阵:\n", metrics.confusion_matrix(Y_test, y_test_pre, labels=[0, 1])
        print "综合报告:\n", metrics.classification_report(Y_test, y_test_pre)
    print 'auc_avg:',np.mean(auc_list)

def rf_model1(anomalys): #
    dataset_len = anomalys.shape[0]
    ratio = 0.8
    train_len = int(ratio*dataset_len)
    X_train = anomalys.iloc[:train_len,:]
    Y_train = X_train.pop('label')
    X_test = anomalys.iloc[train_len:,:]
    Y_test = X_test.pop('label')
    #X_train, X_test, Y_train, Y_test = train_test_split(anomalys, anomalys_label, test_size=0.3)
    print 'Xtrain,Xtest,Ytrain,Ytest: ',X_train.shape,X_test.shape,Y_train.shape,Y_test.shape
    # rf model
    model = RandomForestClassifier(oob_score=True, n_estimators=100, max_features='sqrt')
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

def rf_kf_model():
    res_pathlist = get_res_pathlist()
    anomalys = rf_construct_data(res_pathlist)
    anomalys['timestamp'] = pd.to_datetime(anomalys['timestamp'],unit='s')
    anomalys['timestamp'] = anomalys['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
    anomalys.drop('timestamp',inplace=True,axis=1)
    anomalys_label = anomalys.pop('label')


    scorings = ['precision_macro','recall_macro']
    rf = RandomForestClassifier(oob_score=True, n_estimators=100, max_features='auto')
    score_res = cross_validate(rf,anomalys,anomalys_label,cv=5,scoring=scorings)
    print 'cross res:\n',score_res
    '''
    kf = StratifiedKFold(n_splits=3,shuffle=False)
    auc_list = []
    for train_index,test_index in kf.split(anomalys,anomalys_label):
        X_train,X_test = anomalys.iloc[train_index,:],anomalys.iloc[test_index,:]
        Y_train,Y_test = anomalys_label[train_index],anomalys_label[test_index]

        # rf model
        rf = RandomForestClassifier(oob_score=True,n_estimators=100,max_features=5)
        rf.fit(X_train,Y_train)

        #y_train_preprobs = rf.predict_proba(Y_train)[:,1]
        y_test_pre = rf.predict(X_test)
        y_test_preprobs = rf.predict_proba(X_test)[:,1]

        #auc_train = roc_auc_score(Y_train, y_train_preprobs)
        auc_test = metrics.roc_auc_score(Y_test, y_test_preprobs)
        auc_list.append(auc_test)

        #print "y test prediction:\n", y_test_pre
        print "test auc:\n", auc_test
        print "混淆矩阵:\n", metrics.confusion_matrix(Y_test, y_test_pre,labels=[0,1])
        print "综合报告:\n", metrics.classification_report(Y_test, y_test_pre)
    print 'auc,recall mean: ',np.mean(auc_list)
    '''

def draw_anomalys():
    res_pathlist = get_res_pathlist(ip_dir=evaluate.IP_DIR)
    anomalys = rf_construct_data(res_pathlist)
    #print 'anomalys columns name: ',anomalys.columns
    anomalys['timestamp'] = pd.to_datetime(anomalys['timestamp'],unit='s')
    anomalys['timestamp'] = anomalys['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')

    plt.figure(1, figsize=(20, 8))
    plt.plot(anomalys['timestamp'], anomalys['anomalyscore_relativeEntropy_10.33.208.52_2_4nodes_os_cpu_percent'], label='detected')
    plt.plot(anomalys['timestamp'], anomalys['label'], label='label')
    plt.legend(loc='upper right')
    plt.show()

def main():
    anomalys = load_dataset()
    rf_model1(anomalys)


if __name__ == '__main__':
    main()