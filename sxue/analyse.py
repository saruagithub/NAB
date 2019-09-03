#encoding=utf-8
import pandas as pd
import numpy as np
import datetime
from matplotlib import pyplot as plt
import xlrd
import os


def event_preprocess(events_file='events_20181122084402.xlsx'):
    events = pd.read_excel(events_file)
    return events


def plot_score(res_path):
    '''
    :param res_path: 经过检测的单序列数据路径
    :return: 异常优先级,anomaly_score画在一张图上进行对比
    '''
    events = event_preprocess()
    #events = events[events['IP']=='10.33.208.66']
    events[u'首次发生时间'] = pd.to_datetime(events[u'首次发生时间'],format='%Y-%m-%d %H:%M:%S')
    events = events[events[u'首次发生时间']>datetime.datetime(2018,11,8,0,0,0)]
    events['priority'] = events[u'优先级'].map({u'高':3,u'中':2,u'低':1})
    plt.plot(events[u'首次发生时间'],events['priority'])

    res_one = pd.read_csv(res_path)
    res_one['timestamp'] = pd.to_datetime(res_one['timestamp'], format='%Y-%m-%d %H:%M:%S')
    plt.plot(res_one['timestamp'],res_one['anomaly_score'],color='green')

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
    anomalys = res[res['anomaly_score']>0.5]
    anomalys = pd.concat([anomalys,anomalys['timestamp'].apply(lambda x:find_next_log(x))],axis=1)
    anomalys.to_csv(res_path.rstrip('.csv')+'_log.csv')

def main():
    #parameters
    res_csv_path = '../results/myres/numenta/es_node3_noIP_noAveTime/numenta_2_3nodes_os_cpu_load_average_1m.csv'
    contrast(res_csv_path)
    plot_score(res_csv_path)

if __name__ == '__main__':
    main()