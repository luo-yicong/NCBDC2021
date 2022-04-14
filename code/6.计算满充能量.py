import os
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

#计算箱型图上下界
def limit(data):
    data_sort=dict(sorted(data.items(),key=lambda x:x[1],reverse=False))
    count_25=round(len(data)/4)
    count_75=round(3*len(data)/4)
    i=0
    for key in data_sort.keys():
        if i==count_25:
            key_25=key
        if i==count_75:
            key_75=key
        i+=1
    value_25=data_sort[key_25]
    value_75=data_sort[key_75]
    max_limit=value_75+1.5*(value_75-value_25)
    min_limit=value_25-1.5*(value_75-value_25)
    return max_limit,min_limit

#检测充电且SOC在20%到90%
def detect_charge(data,SOC_low=20,SOC_high=90):
    mileage=-1
    start=-1
    finish=-1
    for row in range(data.shape[0]):
        if data.loc[row,'充电状态']==1 :
            if data.loc[row,'SOC']>SOC_low and data.loc[row,'SOC']<SOC_high:
            #if data.loc[row,'SOC_fit']>SOC_low and data.loc[row,'SOC_fit']<SOC_high:
                if start==-1:
                    start=row
                    mileage=data.loc[start,'累计里程']
                    
                finish=row
    return start,finish,mileage

#计算某时间段满充能量
def calculate_subcapacity(data,start,finish):
    start_temp=start
    current_integral=0
    while(start<finish):
        start+=1
        current_integral+=(data.loc[start,'总电压']+data.loc[start-1,'总电压'])/2*(data.loc[start,'总电流']+data.loc[start-1,'总电流'])/2*(data.loc[start,'时间']-data.loc[start-1,'时间']).total_seconds()/3600
    if data.loc[finish,'SOC']-data.loc[start_temp,'SOC']<2:
        return -1
    capacity=-current_integral/(data.loc[finish,'SOC']-data.loc[start_temp,'SOC'])*100/1000
    return capacity    

#计算满充能量
def calculate_capacity(data):
    data.loc[:,'时间']=pd.to_datetime(data.loc[:,'时间'])
    data=data.reset_index(drop='True')
    start,finish,mileage=detect_charge(data)
    n=finish-start
    if start==-1 or finish==-1 or n<20:
        return mileage,-1
    capacity_list=[]
    for i in range(n-23):
        capacity=calculate_subcapacity(data,start+i+1,start+24+i)
        if capacity!=-1 :
            capacity_list.append(capacity)
    if len(capacity_list)==0:
        return mileage,-1
    capacity=np.mean(capacity_list)
    return mileage,capacity

#线性拟合
def capcity_pred(mileage,capacity):
    mileage=np.array(mileage).reshape(-1,1)
    capacity=np.array(capacity)
    mileage_capacity_model = linear_model.LinearRegression()
    mileage_capacity_model=linear_model.Lasso(alpha=0.5)
    mileage_capacity_model.fit(mileage,capacity)
    return mileage_capacity_model

#计算某辆车满充能量

def calculate(data_charge,dirname):
    mileage_capacity={}
    for charge in data_charge:
        mileage,capacity=calculate_capacity(charge)
        if capacity!=-1 :
            mileage_capacity[mileage]=capacity 
    max_limit,min_limit=limit(mileage_capacity)
    delete_set=set()
    #满充能量异常值处理
    for key in mileage_capacity.keys():
        if mileage_capacity[key]>max_limit or mileage_capacity[key]<min_limit:
            delete_set.add(key)
    for key in delete_set:
        mileage_capacity.pop(key)
    #保存满充能量和里程关系
    pd.DataFrame(mileage_capacity,index=[0]).to_csv(os.path.join('../data2/mileage_capacity',dirname+'.csv'),index=False)
    mileage_capacity_model=capcity_pred(list(mileage_capacity.keys()),list(mileage_capacity.values()))
    #print(dirname,mileage_capacity_model.coef_,mileage_capacity_model.intercept_)
    
    #使用线性模型计算行驶时每一里程对应的满充容量
    dir=os.path.join('../data2/drive_clean_outlier',dirname)
    files=os.listdir(dir)
    for file in files:
        data=pd.read_csv(os.path.join(dir,file))
        if len(data)==0:
            continue
        data['满充能量']=list(mileage_capacity_model.predict(np.array(data['累计里程']).reshape(-1,1)))
        data.to_csv(os.path.join(dir,file),index=False)

    x=np.arange(0,300000).reshape(-1,1)
    y=mileage_capacity_model.predict(x)
    
    '''plt.text(0,y[0],str(y[0]))
    plt.text(299999,y[-1],str(y[-1]))
    plt.scatter(mileage_capacity.keys(),mileage_capacity.values(),s=1)
    plt.plot(x,y,color='g')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.xlabel('累计里程/km')
    plt.ylabel('满充容量/kw·h')
    plt.yticks(np.arange(20,115,5))
    plt.show()'''
    

def main(path):
    dirnames=os.listdir(path)
    for dirname in dirnames:
        data_charge=[]#某辆车所有充电片段
        for filename in os.listdir(os.path.join(path,dirname)):
            data_charge.append(pd.read_csv(os.path.join(path,dirname,filename)))
        calculate(data_charge,dirname)

if __name__=='__main__':
    time_start=time.time()
    path='../data2/charge_clean_outlier'
    main(path)
    time_finish=time.time()
    print('time cost :',time_finish-time_start,'s')