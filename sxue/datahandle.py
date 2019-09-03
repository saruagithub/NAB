import numpy as np
import pandas as pd
import os

def readdata(filepath):
    data =  pd.read_csv(filepath)
    return data

def drop_unname(data):
    data.drop(data.columns[0], axis=1, inplace=True)
    return data

def change_col(data,col_name,index):
    col = data[col_name]
    data.drop(columns=col_name, inplace=True)
    data.insert(index, col_name, col)
    return data

def main():
    # dir = os.getcwd()
    # print dir
    cpu_percent = readdata('/home/andrew/Projects/ecg-htm/mytest/1122bkmoni_selfscript_es_nodes_3_nodes_os_cpu_percent.csv')
    print(cpu_percent.head(10))
    cpu_percent = drop_unname(cpu_percent)
    cpu_percent.to_csv('es_nodes_3_nodes_os_cpu_percent.csv',columns=['time','nodes_os_cpu_percent'],index=0)

if __name__ == '__main__':
    main()