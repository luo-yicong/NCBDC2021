import os
import pandas as pd
import time
import numpy as np
from sklearn.cluster import KMeans
import joblib
#归一化
def feature_norm(data):
    for column in data.columns:
        if column=='能耗' or column =='时间':
            continue
        max_value=data[column].max()
        min_value=data[column].min()
        if max_value==min_value:
            continue
        data[column]=(data[column]-min_value)/(max_value-min_value)
    return data

#特征提取
def feature_extract(data_split,speed_cluster_model):
    sample_feature=[]
    temperature=pd.read_csv('../data2/temperature.csv')
    
    for data in data_split:
        features={}
        temperature_list=[]

        '''for index,row in data.iterrows():
            time_temp=str(data.loc[index,'时间'])
            if int(time_temp[14:16])>=30:
                t=time_temp[:14]+'30'
            else:
                t=time_temp[:14]+'00'
            temperature_list.append(temperature.loc[temperature['时间'] ==t]['气温'])
        
        features['平均气温']=np.mean(temperature_list)'''
        
        
        data.loc[:,'时间']=pd.to_datetime(data.loc[:,'时间'])
        data=data.reset_index()
        data['功率']=data['总电流']*data['总电压']
        data['加速度']=data['车速'].diff()/data['相对时间'].diff()/3.6
        #features['起始时间']=data.loc[0,'时间']
        features['SOC变化']=data.loc[0,'SOC']-data.loc[data.shape[0]-1,'SOC']
        if (features['SOC变化']<0):
            break
        features['总电压变化']=data['总电压'].max()-data['总电压'].min()
        #features['电压差']=data.loc[0,'总电压']-data.loc[data.shape[0]-1,'总电压']
        features['总电压方差']=data['总电压'].var()
        features['最高车速']=data['车速'].max()
        features['总行驶时长']=(data.loc[data.shape[0]-1,'时间']-data.loc[0,'时间']).total_seconds()              
        features['总行驶里程']=data.loc[data.shape[0]-1,'累计里程']-data.loc[0,'累计里程']
        features['平均车速']=features['总行驶里程']/features['总行驶时长']*3600
        #features['车速标准差']=data['车速'].std()
        features['最大瞬时功率']=data['功率'].max()
        features['平均行驶功率']=data['功率'].mean()
        features['行驶功率方差']=data['功率'].var()
        features['平均加速度']=features['平均车速']/features['总行驶时长']/3.6
        #features['平均电流']=data['总电流'].mean()
        #features['总电流方差']=data['总电流'].var()
        
        features['平均单体电压差']=(data['电池单体电压最高值']-data['电池单体电压最低值']).mean()
        #features['最大单体电压差']=(data['电池单体电压最高值']-data['电池单体电压最低值']).max()
        #features['最小单体电压差']=(data['电池单体电压最高值']-data['电池单体电压最低值']).min()
        #features['单体电压差方差']=(data['电池单体电压最高值']-data['电池单体电压最低值']).var()
        features['开始累积里程']=data.loc[0,'累计里程']
        #features['结束里程']=data.loc[data.shape[0]-1,'累计里程']
        #features['开始SOC']=data.loc[0,'SOC']
        #features['结束SOC']=data.loc[data.shape[0]-1,'SOC']
        features['怠速时长']=0
        stop_time=0
        features['低速行驶时长']=0
        features['中速行驶时长']=0
        features['高速行驶时长']=0
        features['起步次数']=0
        features['制动回收时长']=0
        features['加速时长']=0
        features['减速时长']=0
        #features['加速踏板总行程']=0
        for index,row in data.iterrows():
            if data.loc[index,'车速']==0 and data.loc[index,'总电流']!=0:
                if index==0:
                    features['怠速时长']+=(data.loc[1,'相对时间']-data.loc[0,'相对时间'])/2
                elif index==data.shape[0]-1:
                    features['怠速时长']+=(data.loc[data.shape[0]-1,'相对时间']-data.loc[data.shape[0]-2,'相对时间'])/2
                else:
                    features['怠速时长']+=(data.loc[index+1,'相对时间']-data.loc[index-1,'相对时间'])/2
            
            if data.loc[index,'车速']==0 and data.loc[index,'总电流']==0:
                if index==0:
                    stop_time+=(data.loc[1,'相对时间']-data.loc[0,'相对时间'])/2
                elif index==data.shape[0]-1:
                    stop_time+=(data.loc[data.shape[0]-1,'相对时间']-data.loc[data.shape[0]-2,'相对时间'])/2
                else:
                    stop_time+=(data.loc[index+1,'相对时间']-data.loc[index-1,'相对时间'])/2

            if data.loc[index,'车速']!=0 and speed_cluster_model.predict(np.array(data.loc[index,'车速']).reshape(-1,1))==speed_cluster_model.predict(np.array(0).reshape(-1,1)):
                if index==0:
                    features['低速行驶时长']+=(data.loc[1,'相对时间']-data.loc[0,'相对时间'])/2
                elif index==data.shape[0]-1:
                    features['低速行驶时长']+=(data.loc[data.shape[0]-1,'相对时间']-data.loc[data.shape[0]-2,'相对时间'])/2
                else:
                    features['低速行驶时长']+=(data.loc[index+1,'相对时间']-data.loc[index-1,'相对时间'])/2
            
            if speed_cluster_model.predict(np.array(data.loc[index,'车速']).reshape(-1,1))==speed_cluster_model.predict(np.array(20).reshape(-1,1)):
                if index==0:
                    features['中速行驶时长']+=(data.loc[1,'相对时间']-data.loc[0,'相对时间'])/2
                elif index==data.shape[0]-1:
                    features['中速行驶时长']+=(data.loc[data.shape[0]-1,'相对时间']-data.loc[data.shape[0]-2,'相对时间'])/2
                else:
                    features['中速行驶时长']+=(data.loc[index+1,'相对时间']-data.loc[index-1,'相对时间'])/2

            if speed_cluster_model.predict(np.array(data.loc[index,'车速']).reshape(-1,1))==speed_cluster_model.predict(np.array(40).reshape(-1,1)):
                if index==0:
                    features['高速行驶时长']+=(data.loc[1,'相对时间']-data.loc[0,'相对时间'])/2
                elif index==data.shape[0]-1:
                    features['高速行驶时长']+=(data.loc[data.shape[0]-1,'相对时间']-data.loc[data.shape[0]-2,'相对时间'])/2
                else:
                    features['高速行驶时长']+=(data.loc[index+1,'相对时间']-data.loc[index-1,'相对时间'])/2

            if index!=0:
                if data.loc[index-1,'车速']<=5 and data.loc[index,'加速度']>=0.15:
                    features['起步次数']+=1
                if data.loc[index-1,'总电流']<0 and data.loc[index,'加速度']<=-0.15:
                    features['制动回收时长']+=(data.loc[index,'相对时间']-data.loc[index-1,'相对时间'])
                if data.loc[index-1,'车速']>=5  and data.loc[index,'加速度']>=0.15:
                    features['加速时长'] +=(data.loc[index,'相对时间']-data.loc[index-1,'相对时间'])
                if data.loc[index-1,'总电流']>0 and data.loc[index,'加速度']<=-0.15:
                    features['减速时长'] +=(data.loc[index,'相对时间']-data.loc[index-1,'相对时间'])

            '''if index!=0:
                time_diff=data.loc[index,'相对时间']-data.loc[index-1,'相对时间']
                pedal_mean=(data.loc[index-1,'加速踏板行程值']+data.loc[index,'加速踏板行程值'])/2
                if data.loc[index-1,'加速踏板行程值']!=0 and data.loc[index,'加速踏板行程值']!=0:
                    features['加速踏板总行程']+=time_diff*pedal_mean
                if (data.loc[index-1,'加速踏板行程值']==0 and data.loc[index,'加速踏板行程值']!=0) or (data.loc[index-1,'加速踏板行程值']!=0 and data.loc[index,'加速踏板行程值']==0):
                    features['加速踏板总行程']+=time_diff/2*pedal_mean'''
        #features['加速踏板总行程/里程']=features['加速踏板总行程']/features['总行驶里程']
        
        
        features['平均行驶车速']=features['总行驶里程']/(features['总行驶时长']-features['怠速时长']-stop_time)*1000


        features['早高峰时长']=0
        features['晚高峰时长']=0
        if data.loc[0,'时间'].hour<7 :
            if data.loc[data.shape[0]-1,'时间'].hour in range(7,9):
                features['早高峰时长']=(data.loc[data.shape[0]-1,'时间'].hour-7)*3600+data.loc[data.shape[0]-1,'时间'].minute*60+data.loc[data.shape[0]-1,'时间'].second
            elif data.loc[data.shape[0]-1,'时间'].hour>=9:
                features['早高峰时长']=7200
        elif data.loc[0,'时间'].hour in range(7,9):
            if data.loc[data.shape[0]-1,'时间'].hour in range(7,9):
                features['早高峰时长']=(data.loc[data.shape[0]-1,'时间']-data.loc[0,'时间']).total_seconds()
            elif data.loc[data.shape[0]-1,'时间'].hour>=9:
                features['早高峰时长']=7200-((data.loc[0,'时间'].hour-7)*3600+data.loc[0,'时间'].minute*60+data.loc[0,'时间'].second)
        if data.loc[0,'时间'].hour<17 :
            if data.loc[data.shape[0]-1,'时间'].hour in range(17,20):
                features['晚高峰时长']=(data.loc[data.shape[0]-1,'时间'].hour-17)*3600+data.loc[data.shape[0]-1,'时间'].minute*60+data.loc[data.shape[0]-1,'时间'].second
            elif data.loc[data.shape[0]-1,'时间'].hour>=20:
                features['晚高峰时长']=10800
        elif data.loc[0,'时间'].hour in range(17,20):
            if data.loc[data.shape[0]-1,'时间'].hour in range(17,20):
                features['晚高峰时长']=(data.loc[data.shape[0]-1,'时间']-data.loc[0,'时间']).total_seconds()
            elif data.loc[data.shape[0]-1,'时间'].hour>=20:
                features['晚高峰时长']=10800-((data.loc[0,'时间'].hour-17)*3600+data.loc[0,'时间'].minute*60+data.loc[0,'时间'].second)
        features['平均单体温度差']=(data['最高温度值']-data['最低温度值']).mean()
        #features['最大单体温度差']=(data['最高温度值']-data['最低温度值']).max()
        #features['最小单体温度差']=(data['最高温度值']-data['最低温度值']).min()
        features['单体温度差方差']=(data['最高温度值']-data['最低温度值']).var()

        #features['dSOC/d行驶时间']=features['SOC变化']/(features['总行驶时长']-features['怠速时长']-stop_time)
        #features['du/dSOC']=features['总电压变化']/features['SOC变化']
        #features['dSOC/d里程']=features['SOC变化']/features['总行驶里程']
        features['单体最高温度']=data['最高温度值'].max()
        features['单体最低温度']=data['最低温度值'].min()
        #features['加速踏板平均行程值']=data['加速踏板行程值'].mean()
        #features['制动踏板平均状态']=data['制动踏板状态'].mean()
        '''features['春']=0
        features['夏']=0
        features['秋']=0
        features['冬']=0
        if data.loc[0,'时间'].month in range(3,6):
            features['春']=1
        if data.loc[0,'时间'].month in range(6,9):
            features['夏']=1
        if data.loc[0,'时间'].month in range(9,12):
            features['秋']=1
        if data.loc[0,'时间'].month==12 or data.loc[0,'时间'].month in range(1,3):
            features['冬']=1'''
        #features['开始电压']=data.loc[0,'总电压']
        
        features['能耗']=(data.loc[0,'满充能量']*data.loc[0,'SOC']-data.loc[data.shape[0]-1,'满充能量']*data.loc[data.shape[0]-1,'SOC'])/100
        sample_feature.append(features) 
    sample_feature=pd.DataFrame(sample_feature)  
    return sample_feature

#车速聚类，来判别低中高速
def speed_cluster(path):
    dirnames=os.listdir(path)
    speed=[]
    for dirname in dirnames:
        filenames=os.listdir(os.path.join(path,dirname))
        for filename in filenames:
            data=pd.read_csv(os.path.join(path,dirname,filename))
            speed.extend(list(data['车速']))
    x=np.array(speed).reshape(-1,1)
    model=KMeans(n_clusters=3, n_init=20,max_iter=300,random_state=None).fit(x)
    print (model.cluster_centers_)
    return model


def mileage_integration():
    path='../data2/drive_clean_outlier'
    dirnames=os.listdir(path)
    for dirname in dirnames:
        mileage=[]
        filenames=os.listdir(os.path.join(path,dirname))
        for filename in filenames:
            data=pd.read_csv(os.path.join(path,dirname,filename))
            data.loc[:,'时间']=pd.to_datetime(data.loc[:,'时间'])
            if len(data)<=1:
                continue
            mileage_diff=data.loc[data.shape[0]-1,'累计里程']-data.loc[0,'累计里程']
            mileage.append(mileage_diff)
        pd.DataFrame(mileage).to_csv(os.path.join('../data2/mileage/',dirname+'.csv'),index=False,header=False)

#选择行驶里程符合要求的片段
def mileage_roadmap(path):
    if not os.path.exists('../data2/cluster'):
        dirnames=os.listdir(path)
        for dirname in dirnames:
            if dirname=='part-01448-741da358-7624-4bb4-806b-835c106c6b2d.c000' or dirname=='part-01793-741da358-7624-4bb4-806b-835c106c6b2d.c000' :
                low_threshold=33
                high_threshold=34.5
            else:
                low_threshold=34
                high_threshold=35
        
            filenames=os.listdir(os.path.join(path,dirname))
            if not os.path.exists(os.path.join('../data2/cluster',dirname)):
                os.makedirs(os.path.join('../data2/cluster',dirname))
            for filename in filenames:
                data=pd.read_csv(os.path.join(path,dirname,filename))            
                if len(data)<=1:
                    continue
                mileage_diff=data.loc[data.shape[0]-1,'累计里程']-data.loc[0,'累计里程']
                
                if mileage_diff<=high_threshold and mileage_diff >=low_threshold:
                    data.to_csv(os.path.join('../data2/cluster',dirname,filename),index=False)
    return '../data2/cluster'

def main(path):
    #mileage_integration()
    path=mileage_roadmap(path)
    if not os.path.exists('../data2/speed_cluster_model.pkl'):
        speed_cluster_model=speed_cluster(path)
        joblib.dump(speed_cluster_model,'../data2/speed_cluster_model.pkl')
    else:
        speed_cluster_model=joblib.load('../data2/speed_cluster_model.pkl')
    
    #建立特征文件夹
    if not os.path.exists('../data2/feature'):
        os.makedirs('../data2/feature')
    if not os.path.exists('../data2/feature_norm'):
        os.makedirs('../data2/feature_norm')
    
    dirnames=os.listdir(path)
    
    
    for dirname in dirnames:
        filenames=os.listdir(os.path.join(path,dirname))
        data_snippet=[]
        for filename in filenames:
            file=os.path.join(path,dirname,filename)
            data=pd.read_csv(file)
            data.loc[:,'时间']=pd.to_datetime(data.loc[:,'时间'])
            data_snippet.append(data) 
        
        sample_feature=feature_extract(data_snippet,speed_cluster_model)
        print(len(sample_feature))
        #sample_feature.to_csv(os.path.join('../data2/feature',dirname+'.csv'),sep=',',index=False,header=True)
        #sample_feature_norm=feature_norm(sample_feature)
        #sample_feature_norm.to_csv(os.path.join('../data2/feature_norm',dirname+'.csv'),sep=',',index=False,header=True)

if __name__=='__main__':
    time_start=time.time()
    main('../data2/drive_clean_outlier')
    time_finish=time.time()
    print('time cost :',time_finish-time_start,'s')