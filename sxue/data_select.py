# -*- coding:utf-8 -*-
import pandas as pd
import os
import json

#　timestamp & value　保存到csv
def single_sequence(data,cols,save_path,ip='noip'):
    '''
    :param data:从infludb导出的原始数据表
    :param cols:从表里要单独提出来的所有列名
    :param save_path:将单序列数据指标保存,格式为timestamp,value
    :return:保存到save_path里
    '''
    for i in range(len(cols)):
        # get that col and timestamp
        data_col = data[['timestamp',cols[i]]]
        data_col.rename(columns={cols[i]: 'value'}, inplace=True)
        if len(list(set(data_col['value'].values.tolist()))) < 5:
            continue
        data_col.dropna(inplace=True)
        data_col = data_col.groupby('timestamp').mean().reset_index()
        data_col.to_csv(save_path+ip+'_'+'2_'+str(i+1)+cols[i]+'.csv',index=0)


def write_json(path):
    '''
    :param path: 对path下所有要进行指标异常检测的序列遍历
    :return: 将路径保存到/labels/combined_windows.json里去
    '''
    filesDict = {}
    # 读取文件，造造字典
    for item in os.listdir(path):
        try:
            fullpath = path + item
            filesDict[fullpath.lstrip('../data/')] = []
        except OSError, e:
            print e
    # 保存json
    jsonData = json.dumps(filesDict, sort_keys=True, indent=4)
    fileObject = open('../labels/combined_windows.json','w')
    fileObject.write(jsonData)
    fileObject.close()


def main():
    # adjusted paremeters
    data_csv_path = '../data/realBKmonitor/1122bkmonitor_selfscript_es_nodes_3.csv'
    select_ip = True
    nodes_host_ip = ['10.33.208.35', '10.33.208.52', '10.33.208.43', '10.33.208.37', '10.33.208.36', '10.33.208.44',
                     '10.33.208.53', '10.33.208.51', '10.33.208.45']
    single_data_path = '../data/es_nodes3_9ips/'
    cols = ['nodes_fs_total_available_in_b',
            'nodes_jvm_mem_heap_used_per',
            'nodes_os_cpu_load_average_1m',
            'nodes_os_cpu_percent',
            'nodes_process_cpu_percent']

    # read data and preprocess
    data = pd.read_csv(data_csv_path)
    print data.columns
    data.rename(columns={'time':'timestamp'},inplace=True)
    data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y-%m-%dT%H:%M:%SZ')

    if not os.path.exists(single_data_path):
        os.makedirs(single_data_path)
    # 　nodes_host 筛选
    if select_ip:
        for nodes_host in nodes_host_ip:
            data_ip = data[data['nodes_host'] == nodes_host]
            single_sequence(data_ip, cols, single_data_path, nodes_host)
    else:
        single_sequence(data, cols, single_data_path)

    # 写入到windows_combined里
    write_json(single_data_path)




if __name__ == '__main__':
    main()