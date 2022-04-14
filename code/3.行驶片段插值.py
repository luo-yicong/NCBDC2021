#使用插值的方法对行驶片段的缺省值进行填补
import pandas as pd
import os
from scipy.interpolate import interp1d,PchipInterpolator
import time

#插值函数
def interpolate_method(x,y,method):
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
        if column=='车速':
            if fill_value<0:
                fill_value=0
            elif fill_value>data_column.max():
                fill_value=data_column.max()
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
        filenames=os.listdir(os.path.join(path,dirname))
        if not os.path.exists(os.path.join('../data2/drive_clean',dirname)):
            os.makedirs(os.path.join('../data2/drive_clean',dirname))
        for filename in filenames:
            filedir=os.path.join(path,dirname,filename)
            
            data=pd.read_csv(filedir)
            data=interpolate_process(data,'车速','Hermite')
            data=interpolate_process(data,'总电压','Hermite')
            data=interpolate_process(data,'总电流','Hermite')
            data=interpolate_process(data,'SOC','linear')
            data=interpolate_process(data,'电池单体电压最高值','linear')
            data=interpolate_process(data,'电池单体电压最低值','linear')
            data=interpolate_process(data,'最高温度值','linear')
            data=interpolate_process(data,'最低温度值','linear')
            data=interpolate_process(data,'驱动电机转速','linear')
            data=interpolate_process(data,'驱动电机转矩','linear')
            data=interpolate_process(data,'驱动电机温度','linear')
            data=interpolate_process(data,'驱动电机控制器温度','linear')
            data=interpolate_process(data,'电机控制器输入电压','linear')
            data=interpolate_process(data,'电机控制器直流母流电流','linear')
            data=interpolate_process(data,'经度','linear')
            data=interpolate_process(data,'维度','linear')
            data=interpolate_process(data,'加速踏板行程值''linear')
            data=interpolate_process(data,'制动踏板状态','linear')
            #修正加速踏板行程值和制动踏板状态
            for row in range(1,data.shape[0]-1):
                if data.loc[row,'加速踏板行程值']!=0 and data.loc[row,'制动踏板状态']!=0:
                    if data.loc[row,'车速']>data.loc[row-1,'车速']:
                        data.loc[row,'制动踏板状态']=0
                    else:
                        data.loc[row,'加速踏板行程值']=0
            data.to_csv(os.path.join('../data2/drive_clean',dirname,filename),index=False)

if __name__=='__main__':
    path='../data2/drive'
    time_start=time.time()
    main(path)
    time_finish=time.time()
    print('time cost :',time_finish-time_start,'s')