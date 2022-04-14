#使用插值的方法对充电片段的缺省值进行填补
from numpy import NaN
import pandas as pd
import os
from scipy import interpolate
from scipy.interpolate import lagrange,interp1d,splrep,splev,PchipInterpolator

import time

#插值函数
def interpolate_method(x,y,method):
    #interpolate_f=lagrange(x.values,y.values)
    if method=='Hermite':
        interpolate_f=PchipInterpolator(x.values,y.values)
    if method=='linear':
        interpolate_f=interp1d(x.values,y.values,fill_value='extrapolate')
    return interpolate_f

#插值处理
def interpolate_process(data,column,method):
    data_column=data[column]
    NaN_index_list=list(data_column[data_column.isnull()].index)
    data_not_NaN=data[data_column.notnull()]
    if len(data_not_NaN)<=3 :
        return data
   
    interpolate_f=interpolate_method(data_not_NaN['相对时间'],data_not_NaN[column],method)
    for i in range(len(NaN_index_list)):
        NaN_index=NaN_index_list[i]
        fill_value=interpolate_f(data.loc[NaN_index,'相对时间'])
        #插值结果超过范围则对插值结果进行修正
        if column=='总电压' or column=='总电流':
            if fill_value<data_column.min():
                fill_value=data_column.min()
            elif fill_value>data_column.max():
                fill_value=data_column.max()
        data.loc[NaN_index,column]=fill_value
    return data

def main(path):
    
    dirnames=os.listdir(path)
    for dirname in dirnames:
        target_dir=os.path.join('../data2/charge_clean',dirname)
        if not os.path.exists(target_dir): 
            os.makedirs(target_dir)
        filenames=os.listdir(os.path.join(path,dirname,'charge'))
        for filename in filenames:
            
            filedir=os.path.join(path,dirname,'charge',filename)
            data=pd.read_csv(filedir)
            data.loc[:,'时间']=pd.to_datetime(data.loc[:,'时间'])
            if data.shape[0]!=0:
                data['相对时间']=(data['时间']-data.loc[0,'时间']).dt.total_seconds()
            data=interpolate_process(data,'总电压','Hermite')
            data=interpolate_process(data,'总电流','Hermite')
            data=interpolate_process(data,'SOC','linear')
            data.to_csv(os.path.join(target_dir,filename),index=False)

if __name__=='__main__':
    path='../data2/split'
    time_start=time.time()
    main(path)
    time_finish=time.time()
    print('time cost :',time_finish-time_start,'s')