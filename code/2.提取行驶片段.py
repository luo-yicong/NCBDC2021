#从未充电片段提取行驶片段
import pandas as pd
import os
import time


def drive_snippet(data,target_dir,filename):
    data.loc[:,'时间']=pd.to_datetime(data.loc[:,'时间'])
    time_diff=data['时间'].diff().dt.total_seconds()
    #以熄火时间超过15分钟为分割点，选择片段中分割点距离最大的一段
    row_list=list(time_diff[time_diff>900].index)
    row_list.insert(0,0)
    row_list.append(data.shape[0])
    row_series=pd.Series(row_list)
    index=row_series[(row_series.diff())==row_series.diff().max()].index[0]
    start_row,end_row=row_list[index-1],row_list[index]
    data_drive=data[start_row:end_row].reset_index(drop='True')
    
    #删除行驶片段首尾熄火部分
    frow=0
    while(frow<data_drive.shape[0]-1):
        if data_drive.loc[frow,'车辆状态']==1:
            break
        frow+=1
    brow=data_drive.shape[0]-1
    while(brow>0):
        if data_drive.loc[brow,'车辆状态']==1:
            break
        brow-=1
    if frow<brow:
        data_drive=data_drive.loc[frow:brow].reset_index(drop='True')
    if data_drive.shape[0]!=0:
        data_drive['相对时间']=(data_drive['时间']-data_drive.loc[0,'时间']).dt.total_seconds()
    data_drive.to_csv(os.path.join(target_dir,filename),index=False)
    
def main(path):
    dirnames=os.listdir(path)
    for dirname in dirnames:
        target_dir=os.path.join('../data2/drive',dirname)
        if not os.path.exists(target_dir): 
            os.makedirs(target_dir)
        filenames=os.listdir(os.path.join(path,dirname,'discharge'))
        for filename in filenames:
            data=pd.read_csv(os.path.join(path,dirname,'discharge',filename))
            drive_snippet(data,target_dir,filename)

if __name__=='__main__':
    path='../data2/split'
    time_start=time.time()
    main(path)
    time_finish=time.time()
    print('time cost :',time_finish-time_start,'s')