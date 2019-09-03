#encoding=utf-8
import pandas as pd
import numpy as np
import datetime
from matplotlib import pyplot as plt
import xlrd
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics  import classification_report,confusion_matrix


def concat_all_anomaly_csv(path):
    # 经过HTM预测得到的所有异常csv，保留timestamp,anomaly_score并左右拼接起来
    res_datas = []
    for item in os.listdir(path):
        data = pd.read_csv(path+item)
        data.drop(['value','raw_score','label'],axis=1 ,inplace=True)
        data = data.rename(columns = {'anomaly_score':'anomalyscore_'+item.rstrip('.csv')})
        print data.columns
        data = data.set_index(['timestamp'])
        res_datas.append(data)

    all_anomalys = pd.concat(res_datas,axis=1)
    all_anomalys.fillna(method='ffill',inplace=True)
    all_anomalys.reset_index(inplace=True)
    all_anomalys.rename(columns = {'index':'timestamp'},inplace=True)
    # drop duplicated timestamp
    #all_anomalys = all_anomalys.loc[:,~all_anomalys.columns.duplicated()]
    return all_anomalys


#recent_index = 0
def log_label(row,events):
    # 优化从后面开始查找最近时间
    #global recent_index
    #events = events.iloc[0:recent_index]
    # 当前时间后续ｎ分钟内出现日志异常即label=1
    interval = events[u'首次发生时间'] - row['timestamp']
    # 时间在之后10min内出现日志异常
    interval =  interval[interval>0]
    # 加速计算，保存最近的时间点，从之后开始查找,recent_index是离 row['timestamp]最近的时间的index
    #recent_index = interval.shape[0] - 1
    interval.reset_index(drop=True,inplace=True)
    row_notime = row.drop('timestamp')
    # anomaly_score > 0.5 && log anomaly time is later than anomaly_score time in 30min
    #if row_notime[row_notime>0.5].shape[0] == 0:
    #    return 0
    if interval[interval.shape[0]-1] > 1800:
        return 0
    else:
        return 1


def construct_data():
    detected_path = '../results/myres/numenta/es_node3_noIP_noAveTime/'
    detected_anomalys = concat_all_anomaly_csv(detected_path)
    # no ave time数据,对anomaly_score做mean
    detected_anomalys = detected_anomalys.groupby('timestamp').mean().reset_index()
    detected_anomalys['timestamp'] = detected_anomalys['timestamp'].apply(lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))

    events = pd.read_excel('events_20181122084402.xlsx')
    events[u'首次发生时间'] = events[u'首次发生时间'].apply(lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))

    # 当前时间后续ｎ分钟内出现日志异常，并且anomaly_score其中有一个>0.5, 即label=1
    detected_anomalys['label'] = detected_anomalys.apply(lambda row: log_label(row, events),axis=1)
    return detected_anomalys


def main():
    anomalys = construct_data()

    # SVM training
    print "SVM training!"
    anomaly_label = anomalys.pop('label')
    X_train,X_test,y_train,y_test = train_test_split(anomalys,anomaly_label,test_size=0.3)
    #svclassifier = SVC(kernel='rbf',gamma=0.1)
    svclassifier = SVC(kernel='rbf')
    svclassifier.fit(X_train,y_train)

    y_pred = svclassifier.predict(X_test)
    print confusion_matrix(y_test,y_pred)
    print classification_report(y_test,y_pred)




if __name__ == '__main__':
    main()