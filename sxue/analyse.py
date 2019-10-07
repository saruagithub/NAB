#encoding=utf-8
import pandas as pd
import numpy as np
import datetime
from matplotlib import pyplot as plt
import xlrd
import os
import predict

ANOMALY_SCORE = 0.5

def event_preprocess(events_file='events_20181122084402.xlsx',es_ip='10.33.208.66'):
    events = pd.read_excel(events_file)
    events['priority'] = events[u'优先级'].map({u'高': 3, u'中': 2, u'低': 1})
    events[u'首次发生时间'] = pd.to_datetime(events[u'首次发生时间'], format='%Y-%m-%d %H:%M:%S')
    events = events[events['IP'] == es_ip]
    events = events[events[u'首次发生时间'] > datetime.datetime(2018, 11, 8, 0, 0, 0)]
    return events

def plot_value(res_path):
    '''
    :param res_path: csv路径
    :return: 画出原始数据的value值变化
    '''
    plt.figure(1, figsize=(20, 8))
    res = pd.read_csv(res_path)
    res['timestamp'] = pd.to_datetime(res['timestamp'],format='%Y-%m-%d %H:%M:%S')
    res.set_index(['timestamp'],inplace=True)
    res['value'].plot(label='time series value')
    plt.legend(loc='upper right')
    plt.show()

def plot_value_score(res_path):
    '''
    :param res_path: 经过检测的单序列数据路径
    :return: 画图异常分数大于0.5的异常值点
    '''
    plt.figure(1, figsize=(20, 8))
    res = pd.read_csv(res_path)
    res['timestamp'] = pd.to_datetime(res['timestamp'],format='%Y-%m-%d %H:%M:%S')
    res.set_index(['timestamp'],inplace=True)
    res['value'].plot(label='time series value')

    res_temp = res[res['anomaly_score']>ANOMALY_SCORE]
    #plt.scatter(x=res_temp.index,y=res_temp['anomaly_score'],c='red',label='anomaly_score > 0.5 in time series')
    plt.scatter(x=res_temp.index, y=res_temp['value'], c='red', label='value whose anomaly_score > 0.5 in time series')

    plt.legend(loc='upper right')
    plt.show()

def plot_value_events(res_path):
    plt.figure(1, figsize=(20, 8))
    events = event_preprocess(events_file='merge_events.xlsx')
    plt.scatter(x=events[u'首次发生时间'],y=events['priority'],color='blue',label='true event priority')

    res = pd.read_csv(res_path)
    res['timestamp'] = pd.to_datetime(res['timestamp'], format='%Y-%m-%d %H:%M:%S')
    res_temp =  res[res['anomaly_score']>ANOMALY_SCORE]
    plt.plot(res_temp['timestamp'],res_temp['value'],color='red',label='value whose anomaly_score > 0.5 in time series')

    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

def plot_score_events(res_path):
    '''
    :param res_path: 经过检测的单序列数据路径
    :return: 异常优先级　& anomaly_score画在一张图上进行对比
    '''
    plt.figure(1, figsize=(20, 8))
    events = event_preprocess(events_file='merge_events.xlsx')
    plt.scatter(x=events[u'首次发生时间'],y=events['priority'],label='true event priority')

    res_one = pd.read_csv(res_path)
    res_one['timestamp'] = pd.to_datetime(res_one['timestamp'], format='%Y-%m-%d %H:%M:%S')
    plt.plot(res_one['timestamp'],res_one['anomaly_score'],color='green',label='anomaly_score in time series')

    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()


def all_anomalys_plot(detected_path):
    '''
    :param detected_path: 经过检测的结果文件夹，如'../results/myres/numenta/es_nodes3_66/'
    :return:score超过ANOMALY_SCORE的个数　与　events事件
    '''
    plt.figure(1,figsize=(20,8))
    anomalys = predict.concat_all_anomaly_csv(detected_path)
    anomalys['count'] = anomalys.apply(lambda row:sum(row>ANOMALY_SCORE)-1,axis=1) #score超过ANOMALY_SCORE的个数
    #anomalys['max_anomaly_score'] = anomalys.iloc[:,1:anomalys.shape[1]-1].max(axis=1) #画一行里的最大值
    anomalys['timestamp'] = pd.to_datetime(anomalys['timestamp'], format='%Y-%m-%d %H:%M:%S')
    plt.plot(anomalys['timestamp'],anomalys['count'],label='score>threshold\'s count after concat_all_anomaly_csv')

    events = event_preprocess(events_file='merge_events.xlsx')
    plt.plot(events[u'首次发生时间'],events['priority'],label='true event priority')

    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

def find_next_log(x):
    '''
    :param x:经过检测的单序列数据的一行的timestamp
    :return:如果单序列数据指标的时间前后两小时内有日志(events)异常,则log_index和log_content存入经过检测的单序列数据中
    '''
    #events = pd.read_excel('events_20181122084402.xlsx')
    events = pd.read_excel('merge_events.xlsx')

    events[u'首次发生时间'] = pd.to_datetime(events[u'首次发生时间'],format='%Y-%m-%d %H:%M:%S')
    interval = np.abs(events[u'首次发生时间'] - x)
    index = np.argmin(interval)
    if interval[index] < pd.Timedelta('0 days 02:00:00'):
        return pd.Series({'log_index':index,'log_content':events.at[index,u'事件标题']})
    else:
        return pd.Series({'log_index':-1,'log_content':''})


def contrast(res_path):
    '''
    :param res_path: 经过检测的单序列数据路径
    :return:查看anomaly_score大于0.5的异常情况和对应的日志异常
    '''
    res = pd.read_csv(res_path)
    res = res.groupby('timestamp').mean().reset_index()
    res['timestamp'] = pd.to_datetime(res['timestamp'])
    anomalys = res[res['anomaly_score']>ANOMALY_SCORE]
    anomalys = pd.concat([anomalys,anomalys['timestamp'].apply(lambda x:find_next_log(x))],axis=1)
    anomalys.to_csv(res_path.rstrip('.csv')+'_log.csv')


def main():
    plot_value('../results/myres/numenta/es_nodes3_66/numenta_10.33.208.66_2_1nodes_os_cpu_load_average_1m.csv')
    plot_value_score('../results/myres/numenta/es_nodes3_66/numenta_10.33.208.66_2_1nodes_os_cpu_load_average_1m.csv')
    plot_value_events('../results/myres/numenta/es_nodes3_66/numenta_10.33.208.66_2_1nodes_os_cpu_load_average_1m.csv')
    plot_score_events('../results/myres/numenta/es_nodes3_66/numenta_10.33.208.66_2_1nodes_os_cpu_load_average_1m.csv')
    #all_anomalys_plot('../results/myres/bayesChangePt/es_nodes3_66/')  #detected_path = '../results/myres/numenta/es_nodes3_9ips/'



if __name__ == '__main__':
    main()