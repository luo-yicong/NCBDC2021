
# 完成车辆状态、充电状态、累计里程的缺省值的填补，并根据充电状态划分充电片段和未充电片段
import pandas as pd
import os
import time

#时间排序
def sort(data):
    data.loc[:,'时间']=pd.to_datetime(data.loc[:,'时间'])
    data_sorted=data.sort_values(by='时间')
    data_sorted.drop_duplicates(subset='时间',keep='first',inplace=True)
    return data_sorted.reset_index(drop='True')

#填补车辆状态缺省值
def vehicle_state_process(data):
    data['车辆状态'].fillna(method='ffill',inplace=True)
    return data

#填补累计里程和异常值处理
def mileage_process(data):
    data['累计里程'].interpolate(method='linear',inplace=True)
    row=1
    while(row<data.shape[0]-1):
        if data.loc[row,'累计里程']<data.loc[row-1,'累计里程'] or data.loc[row,'累计里程']>data.loc[row+1,'累计里程'] :
           data.loc[row,'累计里程']=(data.loc[row-1,'累计里程']+data.loc[row+1,'累计里程'])/2
        row+=1
    return data

#填补充电状态缺省值和异常值处理
def charge_status_process(data):
    data=sort(data)
    #补缺省值
    data['充电状态'].fillna(method='ffill',inplace=True)
    #异常值处理
    row=1
    while(row<data.shape[0]-1):
        if data.loc[row,'充电状态']!=data.loc[row-1,'充电状态'] and data.loc[row,'充电状态']!=data.loc[row+1,'充电状态'] and data.loc[row-1,'充电状态']==data.loc[row+1,'充电状态']:
            data.loc[row,'充电状态']=data.loc[row-1,'充电状态']
        row+=1
    return data

#分割充电片段和未充电片段
def split(data,filename):
    #建立文件夹
    charge_directory=os.path.join('../data2/split',os.path.splitext(filename)[0],'charge')
    discharge_directory=os.path.join('../data2/split',os.path.splitext(filename)[0],'discharge')
    if not os.path.exists(charge_directory): 
        os.makedirs(charge_directory)
    if not os.path.exists(discharge_directory): 
        os.makedirs(discharge_directory)
    
    #记录状态，True充电，False未充电
    first_state=True  
    #记录当前状态，True充电，False未充电
    present_state=True   
    #根据状态之间的变化进行分割
    if data.loc[0,'充电状态']==3:
        first_state=False
    charge_count=1
    discharge_count=1
    first_row=0
    row=1
    while(row<data.shape[0]):
        if data.loc[row,'充电状态']==3:
            present_state=False
        else:
            present_state=True
        if present_state!=first_state:
            data_snippet=data.loc[first_row:row-1]
            data_snippet=data_snippet.reset_index(drop='True')
            if first_state==True:             
                data_snippet.to_csv(os.path.join(charge_directory,str(charge_count)+'.csv'),index=False)
                charge_count+=1
            else:
                data_snippet.to_csv(os.path.join(discharge_directory,str(discharge_count)+'.csv'),index=False)
                discharge_count+=1
            first_state=present_state
            first_row=row
        row+=1
    data_snippet=data.loc[first_row:row-1]
    data_snippet=data_snippet.reset_index(drop='True')
    if first_state==True:                
        data_snippet.to_csv(os.path.join(charge_directory,str(charge_count)+'.csv'),index=False)  
    else:
        data_snippet.to_csv(os.path.join(discharge_directory,str(discharge_count)+'.csv'),index=False)

def main(path):
    filenames=os.listdir(path)
    for filename in filenames:
        filedir=os.path.join(path,filename)
        data=pd.read_csv(filedir)
        #充电状态处理
        data=charge_status_process(data)
        #车辆状态处理
        data=vehicle_state_process(data)
        #累计里程处理
        data=mileage_process(data)
        split(data,filename)

if __name__=='__main__':
    path='../data2/original'
    time_start=time.time()
    main(path)
    time_finish=time.time()
    print('time cost :',time_finish-time_start,'s')