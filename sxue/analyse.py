#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import datetime
from matplotlib import pyplot as plt
import xlrd
import re
import predict

ANOMALY_SCORE = 0.5

def event_preprocess(events_file='events_20181122084402.xlsx', es_ip='10.33.208.66'):
    events = pd.read_excel(events_file)
    events['priority'] = events[u'优先级'].map({u'高': 3, u'中': 2, u'低': 1})
    events[u'首次发生时间'] = pd.to_datetime(events[u'首次发生时间'], format='%Y-%m-%d %H:%M:%S')
    events = events[events['IP'] == es_ip]
    events = events[events[u'首次发生时间'] > datetime.datetime(2018, 11, 8, 0, 0, 0)]
    return events

def res_preprocess(res_path):
    res = pd.read_csv(res_path)
    res['timestamp'] = pd.to_datetime(res['timestamp'], format='%Y-%m-%d %H:%M:%S')
    res_ip = re.findall('(?<![\.\d])(?:\d{1,3}\.){3}\d{1,3}(?![\.\d])',res_path,re.S)
    return res,res_ip[0]

def plot_value_perweek(res):
    pass
    # plt.figure(1, figsize=(20, 8))
    # res['week'] = res['timestamp'].isocalendar()[1]
    # plt.plot(res['week'], res['value'], label='time series value per week')
    # plt.legend(loc='upper right')
    # plt.show()

def plot_value_recentday(res):
    pass

def plot_value(res):
    '''
    :param res_path: csv路径
    :return: 画出原始数据的value值变化
    '''
    plt.figure(1, figsize=(20, 8))
    plt.plot(res['timestamp'], res['value'], label='nodes memory used per minute')
    plt.legend(loc='upper right')
    plt.show()

def plot_value_score(res):
    '''
    :param res_path: 经过检测的单序列数据路径
    :return: 画图异常分数大于0.5的异常值点
    '''
    plt.figure(1, figsize=(20, 8))
    plt.plot(res['timestamp'], res['value'], label='time series value')
    res_temp = res[res['anomaly_score']>ANOMALY_SCORE]
    plt.scatter(x=res_temp['timestamp'], y=res_temp['value'], c='red', label='value whose anomaly_score > 0.5 in time series')
    plt.legend(loc='upper right')
    plt.show()

def plot_value_score_events(res,res_ip):
    '''
    :param res: 经过检测的单序列数据路径
    :return: 画图时序值,　events点,　异常分数>Thld的点
    '''
    plt.figure(1, figsize=(20, 8))
    events = event_preprocess(events_file='merge_events.xlsx',es_ip=res_ip)
    plt.plot(res['timestamp'], res['value'], label='time series value')
    plt.scatter(x=events[u'首次发生时间'],
                y=[res['value'].mean() for x in range(events.shape[0])],
                color='red',
                label='true event priority') #!!! 事件优先级 priority

    res_temp = res[res['anomaly_score'] > ANOMALY_SCORE]
    plt.scatter(x=res_temp['timestamp'],
                y=res_temp['value'],
                c='purple',
                label='value whose anomaly_score > 0.5 in time series')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

def plot_score_events(res):
    '''
    :param res_path: 经过检测的单序列数据路径
    :return: 异常优先级　& anomaly_score画在一张图上进行对比
    '''
    plt.figure(1, figsize=(20, 8))
    events = event_preprocess(events_file='merge_events.xlsx')
    plt.scatter(x=events[u'首次发生时间'],y=events['priority'],label='true event priority')
    plt.plot(res['timestamp'],res['anomaly_score'],color='red',label='anomaly_score in time series')
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
    path = '../results/myres/numenta/es_nodes3_66/numenta_10.33.208.66_2_2nodes_jvm_mem_heap_used_per.csv'
    res,res_ip = res_preprocess(path)
    #plot_value_score_events(res,res_ip)
    plot_value(res)
    #plot_value_perweek(res)
    #plot_value_score(res)
    #plot_score_events(res)


if __name__ == '__main__':
    main()