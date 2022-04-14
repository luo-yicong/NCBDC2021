import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn import metrics
import lightgbm as lgb
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor 
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.inspection import plot_partial_dependence

kf = KFold(n_splits = 5,shuffle=True)
MSE=[]
RMSE=[]
MAE=[]

def print_error(MSE,RMSE,MAE):
    print('MSE:',MSE)
    print('RMSE:',RMSE)
    print('MAE:',MAE)
    print('MSE:',np.mean(MSE))
    print('RMSE:',np.mean(RMSE))
    print('MAE:',np.mean(MAE))
    MSE.clear()
    RMSE.clear()
    MAE.clear()

def append_error(y_test,y_pred):
    MSE_temp=metrics.mean_squared_error(y_test,y_pred)
    MSE.append(MSE_temp)
    RMSE_temp=metrics.mean_squared_error(y_test,y_pred)**0.5
    RMSE.append(RMSE_temp)
    MAE_temp=metrics.mean_absolute_error(y_test,y_pred)
    MAE.append(MAE_temp)
    
def draw(y_pred,y_test):
    #plt.plot(range(len(y_test)),y_test,color='b',marker='o',label='实际值')
    #plt.plot(range(len(y_test)),y_pred,color='r',marker='*',label='预测值')
    plt.scatter(range(len(y_test)),y_test,color='b',marker='o',label='实际值')
    plt.scatter(range(len(y_test)),y_pred,color='r',marker='*',label='预测值')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.legend()
    plt.show()

def gbdt_model(x_train,y_train,x_test,y_test):
    model=GradientBoostingRegressor(n_estimators=10000,max_depth=30)
    model.fit(x_train, y_train)
    return model

def xgb_model(x_train,y_train,x_test,y_test):
    model = xgb.XGBRegressor(max_depth=50, learning_rate=0.01, n_estimators=10000, min_child_weight=20, 
                    seed=0, subsample=1, colsample_bytree=1, gamma=0.8, reg_alpha=0, reg_lambda=0,
                    objective='reg:squarederror')        
    model.fit(x_train, y_train,eval_metric='rmse', eval_set=[(x_train, y_train)], verbose=True, early_stopping_rounds=50)
    return model

def lgb_model(x_train,y_train,x_test,y_test):
    model = lgb.LGBMRegressor(objective='regression', num_leaves=100,max_depth=20,learning_rate=0.01, n_estimators=20000)       
    model.fit(x_train, y_train,eval_metric='rmse', eval_set=[(x_train, y_train)], verbose=True, early_stopping_rounds=100)
    return model

def svr_model(x_train,y_train,x_test,y_test):
    model=SVR(C=1.0, cache_size=2000, coef0=0.0, degree=3, epsilon=0.1,gamma=0.6, kernel='rbf', max_iter=-1, shrinking=True,tol=0.001, verbose=False)
    model.fit(x_train,y_train)
    return model

def knn_model(x_train,y_train,x_test,y_test):
    model=KNeighborsRegressor(n_neighbors=7,weights = 'uniform')
    model.fit(x_train,y_train)
    return model

def dt_model(x_train,y_train,x_test,y_test):
    model=DecisionTreeRegressor()
    model.fit(x_train,y_train)
    return model

def mlp_model(x_train,y_train,x_test,y_test):
    model=MLPRegressor( hidden_layer_sizes=(20,20),  activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=10000, shuffle=True,
    random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False,beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.fit(x_train,y_train)
    return model

def rf_model(x_train,y_train,x_test,y_test):
    model=RandomForestRegressor()
    model.fit(x_train,y_train)
    return model

def plot_lgb_impotance(model):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    lgb.plot_importance(model,grid=False,height=0.5,ignore_zero=False)
    

def feature_norm(data):
    

    for column in data.columns:
        if column!='能耗' and column !='起始时间':
            max_value=data[column].max()
            min_value=data[column].min()
            if max_value==min_value:
                continue
            data[column]=(data[column]-min_value)/(max_value-min_value)
    return data

#所有数据混合
def predict_all_mix(path,modelname):
    files=os.listdir(path)
    data=pd.DataFrame()
    for file in files:
        data_concat=pd.concat([data,pd.read_csv(os.path.join(path,file))])
    data_concat=feature_norm(data_concat)
    #data_concat=pd.DataFrame()
    '''filenames=os.listdir(path)
    for filename in filenames:
        data=pd.read_csv(os.path.join(path,filename),encoding='gbk')
        data_concat=pd.concat([data_concat,data])
    data_concat=data_concat.reset_index(drop=True)'''
    x=data_concat.iloc[:,:-1]
    y=data_concat.iloc[:,-1]
    for train_index, test_index in kf.split(x):
        x_train,x_test=x.loc[train_index],x.loc[test_index]
        y_train,y_test=y.loc[train_index],y.loc[test_index]
       
        model=modelname(x_train,y_train,x_test,y_test)
        append_error(y_test,model.predict(x_test))

        
        #plot_lgb_impotance(model)
        #plt.show()
        #plot_partial_dependence(model,x_train,features=[0, 1])
        #plt.show()
        #xgb.plot_importance(model,grid=False)
        
        
        #draw(y_test,model.predict(x_test))
    print_error(MSE,RMSE,MAE)

#8辆车用来训练
def predict_all(path,modelname):
    data=[]

    data_train=pd.DataFrame()
    data_test=pd.DataFrame()
    filenames=os.listdir(path)
    for filename in filenames:
        data.append(pd.read_csv(os.path.join(path,filename)))
    for train_index, test_index in kf.split(filenames):
        
        for i in range(len(filenames)):
            if i in train_index:
                data_train=pd.concat([data_train,data[i]])
            else:
                data_test=pd.concat([data_test,data[i]])
        data_train=data_train.reset_index(drop=True)
        data_test=data_test.reset_index(drop=True)
        x_train,y_train=data_train.iloc[:,:-1],data_train.iloc[:,-1]
        x_test,y_test=data_test.iloc[:,:-1],data_test.iloc[:,-1]
        
        model=modelname(x_train,y_train,x_test,y_test)
        append_error(y_test,model.predict(x_test))
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        #xgb.plot_importance(model,show_values = False,grid=False,height=0.5)
        plt.show()

    print_error(MSE,RMSE,MAE)

#每辆车单独建模
def predict_each(path,modelname):
    filenames=os.listdir(path)
    result={}
    for filename in filenames:
        data=pd.read_csv(os.path.join(path,filename))
        x=data.iloc[:,1:-1]
        y=data.iloc[:,-1]
        for train_index, test_index in kf.split(x):
            x_train,x_test=x.loc[train_index],x.loc[test_index]
            y_train,y_test=y.loc[train_index],y.loc[test_index]
            
            model=modelname(x_train,y_train,x_test,y_test)
            append_error(y_test,model.predict(x_test))

            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
            plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
            #lgb.plot_importance(model,grid=False,height=0.5)
            plt.show()
        result[filename]=[np.mean(MSE),np.mean(RMSE),np.mean(MAE)]
        print_error(MSE,RMSE,MAE)
        
    print(result)

if __name__=='__main__':
    path='../data2/backup/10/feature'
    modelname=lgb_model
    predict_all_mix (path,modelname)