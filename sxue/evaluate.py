#encoding=utf-8
import pandas as pd
import numpy as np
import predict
import analyse
import time

def get_events(events_path):
    events = pd.read_excel(events_path)
    events = events[events['IP'] == '10.33.208.66'] # 选取主节点
    events[u'首次发生时间'] = events[u'首次发生时间'].apply(lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))
    events.drop_duplicates(subset=[u'首次发生时间'],keep='first',inplace=True)
    return events

def get_all_res(res_dir_path):
    anomalys = predict.concat_all_anomaly_csv(res_dir_path)
    anomalys['timestamp'] = anomalys['timestamp'].apply(lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))
    anomalys['count'] = anomalys.apply(lambda row: sum(row > analyse.ANOMALY_SCORE) - 1, axis=1)  # score超过ANOMALY_SCORE的个数
    anomalys['label'] = anomalys['count'].apply(lambda x:1 if x > 0 else 0)
    return anomalys

def is_detect(row,anomalys):
    # 当前event时间前ｎ分钟内出现日志异常即is_detect=1
    interval = row[u'首次发生时间'] - anomalys['timestamp']
    interval = interval[interval > 0]
    interval.reset_index(drop=True, inplace=True)
    if interval[interval.shape[0] - 1] < 7200:  # 最小的时间间隔大于30min(1800s) 1h
        return 1
    else:
        return 0

def is_FP(row,events):
    interval = row['timestamp'] - events[u'首次发生时间']
    interval = interval[interval < 0]
    interval.reset_index(drop=True, inplace=True)
    if interval[interval.shape[0] - 1] > -7200:  # 最小的时间间隔大于30min(1800s) 1h
        return 1
    else:
        return 0

def main():
    events = get_events('merge_events.xlsx')
    anomalys = get_all_res('../results/myres/numenta/es_nodes3_66/')
    print '检测的异常,　真实异常: ',anomalys[anomalys['label']==1].shape,events.shape

    print '计算TP,FN,FP'
    anomalys_1 = anomalys[anomalys['label']==1]
    events['isDetected'] = events.apply(lambda row:is_detect(row,anomalys_1),axis=1) # event的时间和anomalys的label为１的比
    TP = events['isDetected'].sum()
    print 'TP: ',TP

    anomalys_1['isFP'] = anomalys_1.apply(lambda row:is_FP(row,events),axis=1)
    FP = anomalys_1['isFP'].sum()
    print 'FP: ',FP

    FN = events.shape[0] - TP
    print 'FN: ',FN

    print 'precision,recall: ',TP/float(TP+FP),TP/float(TP+FN)



if __name__ == '__main__':
    main()