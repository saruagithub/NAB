#encoding=utf-8
import pandas as pd
import time
import os
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.svm import SVC
from sklearn.metrics  import classification_report,confusion_matrix


def concat_all_anomaly_csv(path):
    '''
    :param path: 将path下的所有异常csv结合到一起,timestamp保留一个
    :return: timestamp,各序列异常分数的csv
    '''
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


def log_label(row,events):
    '''
    :param row:每一行
    :param events:日志异常excel表
    :return:如果在当前时间的后n分钟内出现日志异常即label=1
    '''
    # 当前时间后续ｎ分钟内出现日志异常即label=1
    interval = events[u'首次发生时间'] - row['timestamp']
    # 时间在之后10min内出现日志异常
    interval =  interval[interval>0]
    interval.reset_index(drop=True,inplace=True)
    row_notime = row.drop('timestamp')
    # anomaly_score > 0.5 && log anomaly time is later than anomaly_score time in 30min
    if row_notime[row_notime>0.5].shape[0] == 0:
        return 0
    if interval[interval.shape[0]-1] > 1800:
        return 0
    else:
        return 1


def construct_data(detected_path,events_path):
    '''
    :param detected_path: myres/下面的经过异常检测的文件夹路径
    :param events_path: 日志异常的excel路径
    :return: 构建带label的整个数据集
    '''
    detected_anomalys = concat_all_anomaly_csv(detected_path)
    # no ave time数据,对anomaly_score做mean
    detected_anomalys = detected_anomalys.groupby('timestamp').mean().reset_index()
    detected_anomalys['timestamp'] = detected_anomalys['timestamp'].apply(lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))
    events = pd.read_excel(events_path)
    events[u'首次发生时间'] = events[u'首次发生时间'].apply(lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))

    # 当前时间后续ｎ分钟内出现日志异常，并且anomaly_score其中有一个>0.5, 即label=1
    detected_anomalys['label'] = detected_anomalys.apply(lambda row: log_label(row, events),axis=1)
    return detected_anomalys


def main():
    detected_path = '../results/myres/numenta/es_node3_IP/'
    #events_path = 'events_20181122084402.xlsx'
    events_path = 'merge_events.xlsx'
    anomalys = construct_data(detected_path,events_path)

    # SVM training
    print "SVM training!"
    anomaly_label = anomalys.pop('label')
    X_train,X_test,y_train,y_test = train_test_split(anomalys,anomaly_label,test_size=0.2,random_state=8)
    svclassifier = SVC(kernel='rbf',gamma=0.1)
    #svclassifier = SVC(kernel='rbf')
    svclassifier.fit(X_train,y_train)
    #scores = cross_val_score(svclassifier,anomalys,anomaly_label,cv=5)

    y_pred = svclassifier.predict(X_test)
    print "混淆矩阵:\n",confusion_matrix(y_test,y_pred)
    print "综合报告:\n",classification_report(y_test,y_pred)
    #print "Cross vali scores:",scores




if __name__ == '__main__':
    main()