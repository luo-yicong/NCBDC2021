#将片段进行合并，提取箱型图上下限，进行异常值处理
import pandas as pd
import os
import time

#根据时间排序
def sort(data):
    data.loc[:,'时间']=pd.to_datetime(data.loc[:,'时间'])
    data_sorted=data.sort_values(by='时间')
    data_sorted.drop_duplicates(subset='时间',keep='first',inplace=True)
    return data_sorted.reset_index(drop='True')

#合并
def merge(path,target_dir):
    dirnames=os.listdir(path)
    for dirname in dirnames:
        filenames=os.listdir(os.path.join(path,dirname))
        data=pd.DataFrame()
        for filename in filenames:
            data=pd.concat([data,pd.read_csv(os.path.join(path,dirname,filename))],ignore_index=True)
        data=sort(data)
        data.to_csv(os.path.join(target_dir,dirname+'.csv'),index=False)

#计算箱型图上下界
def box_threshold(data,column):
    statistics = data[column].describe() #保存基本统计量
    IQR = statistics.loc['75%']-statistics.loc['25%']   #四分位数间距
    QL = statistics.loc['25%']  #下四分位数
    QU = statistics.loc['75%']  #上四分位数
    threshold1 = QL - 1.5 * IQR #下阈值
    threshold2 = QU + 1.5 * IQR #上阈值
    return threshold1,threshold2 

#异常值处理实现
def outlier_process_detail(data_merge,data,state):
    if state=='drive':
        outlier_column_list=['车速','总电压','经度','维度']
    elif state=='charge':
        outlier_column_list=['总电流','总电压']
    for column in outlier_column_list:
        threshold1,threshold2=box_threshold(data_merge,column)
        data=data.drop(index=list(data[(data[column]<threshold1) | (data[column]>threshold2)].index)).reset_index(drop='True')    
    return data

#各片段分别进行异常处理
def outlier_process(merge_dir,clean_dir,clean_outlier_dir):
    filenames=os.listdir(merge_dir)
    if 'drive' in merge_dir:
        state='drive'
    else:
        state='charge'
    for filename in filenames:
        data_merge=pd.read_csv(os.path.join(merge_dir,filename))
        files=os.listdir(os.path.join(clean_dir,os.path.splitext(filename)[0]))
        if not os.path.exists(os.path.join(clean_outlier_dir,os.path.splitext(filename)[0])):
            os.makedirs(os.path.join(clean_outlier_dir,os.path.splitext(filename)[0]))
        for file in files:
            data=pd.read_csv(os.path.join(clean_dir,os.path.splitext(filename)[0],file))
            data=outlier_process_detail(data_merge,data,state)
            data.to_csv(os.path.join(clean_outlier_dir,os.path.splitext(filename)[0],file),index=False)

def main():
    #合并行驶片段
    path_drive='../data2/drive_clean'
    drive_merge_dir='../data2/drive_merge'
    if not os.path.exists(drive_merge_dir):
        os.makedirs(drive_merge_dir)
    drive_clean_outlier='../data2/drive_clean_outlier'
    merge(path_drive,drive_merge_dir)
    #合并充电片段
    path_charge='../data2/charge_clean'
    charge_merge_dir='../data2/charge_merge'
    if not os.path.exists(charge_merge_dir):
        os.makedirs(charge_merge_dir)
    charge_clean_outlier='../data2/charge_clean_outlier'
    merge(path_charge,charge_merge_dir)

    #异常值处理
    outlier_process(drive_merge_dir,path_drive,drive_clean_outlier)
    outlier_process(charge_merge_dir,path_charge,charge_clean_outlier)

if __name__=='__main__':
    time_start=time.time()
    main()
    time_finish=time.time()
    print('time cost :',time_finish-time_start,'s')

    









   
