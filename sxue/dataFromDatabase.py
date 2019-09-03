import argparse
import pandas as pd
import csv
from influxdb import InfluxDBClient
from openpyxl.workbook import Workbook


def write_pd_csv(influxdb_info,measurement,filename): #field=None,
	print("write points to csv...")
	myclient = InfluxDBClient('localhost', 8086, influxdb_info['user'], influxdb_info['password'], influxdb_info['dbname'])
	points_df = pd.DataFrame(
		#myclient.query('select ' + field +' from ' + measurement + ' order by time ASC;').get_points()
		myclient.query('select * from ' + measurement + ' order by time ASC;').get_points()
	)
	points_df.to_csv(filename,sep=",",index=0)


if __name__ == '__main__':
	influxdbInfo = {'user': 'wangxue', 'password': 'wangxue111111', 'dbname': 'selfscript_3', 'protocol': 'json'}
	influx_measurements = ['selfscript_es_cluster_3','selfscript_es_nodes_3', 'selfscript_jstorm_cluster_3' ,'selfscript_jstorm_componet_3',
	 					   'selfscript_jstorm_graph_3', 'selfscript_jstorm_topology_3', 'selfscript_logstash_monitor_3','selfscript_logstash_syslog_3',
	 					   'selfscript_prekafka_3', 'selfscript_syslog_3' ]
	#fields = ['nodes_os_cpu_percent']
	for measure in influx_measurements:
		#write_pd_csv( influxdbInfo,measure,fields[0],'../data/realBKmonitor/1122bkmoni_'+measure+'.csv')
		write_pd_csv( influxdb_info =influxdbInfo,measurement =measure,
					  filename='../data/realBKmonitor/1122bkmonitor_'+measure+'.csv')
		#write_pd_excel(influxdb_info, '1122bkmoni ' + measurement + '.xlsx', measurement)