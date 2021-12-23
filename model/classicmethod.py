# 1. 导入相关的包
import pandas as pd
import numpy as np
import torch
from collections import defaultdict
from sklearn.preprocessing import StandardScaler,OneHotEncoder,PolynomialFeatures
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor

from random import shuffle
from config.utils import *
from data.SQL import *
from data.Data_copy import *



def dataprepare():
    # 数据准备
    ds_ends = get_date_lista(begin_date='2020-11-01',end_date='2021-02-05') # 后期可以直接transplit
    shuffle(ds_ends) # 不打乱会出现20次为周期的下降
    ds_ends.remove('2020-12-20') # 本来在error里的，减少IO
    ds_ends.remove('2020-12-06') # 本来在error里的，减少IO
    ds_ends.remove('2020-12-13')  # 本来在error里的，减少IO
    ds_ends.remove('2020-12-27') # 本来在error里的，减少IO
    ds_ends.remove('2020-11-22') # 本来在error里的，减少IO
    ds_ends.remove('2020-11-01') # 本来在error里的，减少IO
    label_stabs = torch.Tensor([])
    label_repus = torch.Tensor([])
    label_acts  = torch.Tensor([])
    label_commus = torch.Tensor([])
    label_resos = torch.Tensor([])
    label_abilitys = torch.Tensor([])
    xs = torch.Tensor([])
    for ds_end in ds_ends:
        print(ds_end)
        data_guild, x_catgs,x_conts,wide_dim,dense_dim,embed_dim,\
                   label_stab,label_repu,label_act,label_commu,label_reso,label_ability  = getdatabench(ds_end)
        x_catgs = torch.cat([i for i in x_catgs],dim = 1) # 七天直接concat
        x_conts = torch.cat([i for i in x_conts],dim = 1)
        x = torch.cat([x_catgs,x_conts],dim = 1).cpu() # 总特征
        xs = torch.cat([xs,x],dim = 0) # 合并为总的
        label_stabs = torch.cat([label_stabs,label_stab.cpu()],dim = 0)
        label_repus = torch.cat([label_repus,label_repu.cpu()],dim = 0)
        label_acts = torch.cat([label_acts,label_act.cpu()],dim = 0)
        label_commus = torch.cat([label_commus,label_commu.cpu()],dim = 0)
        label_resos = torch.cat([label_resos,label_reso.cpu()],dim = 0)
        label_abilitys = torch.cat([label_abilitys,label_ability.cpu()],dim = 0)
    # 转换为numpy
    xs = xs.numpy()
    label_stabs = label_stabs.view(-1,).numpy()
    label_repus = label_repus.view(-1,).numpy()
    label_acts = label_acts.view(-1,).numpy()
    label_commus = label_commus.view(-1,).numpy()
    label_resos =label_resos.view(-1,).numpy()
    label_abilitys = label_abilitys.view(-1,).numpy()
    print(xs.shape)  # 直接torch.cat(dim = 0)，应该行数是一样多才行！
    print(label_stabs.shape)  # 如果能够保证label维度和特征维度相合的话，可以加一个for循环！！！遍历所有的ds_end.然后一次存成csv，这个没得图，其实是方便的
    return xs,label_stabs,label_repus,label_acts,label_commus,label_resos,label_abilitys





# 超参数搜索
def gridsearch(train_x,train_y,model,param_grid,n_jobs =-1,verbose = 1,scoring = 'neg_mean_squared_error'):
    '''
    网格搜索调参
    '''
    model= model
    param_grid = param_grid
    scoring = scoring
    grid_search = GridSearchCV(model,param_grid,n_jobs=n_jobs,verbose = verbose,scoring = scoring) #默认5折交叉
    grid_search.fit(train_x,train_y)
    best_params = grid_search.best_estimator_.get_params()
    print("Best params: ")
    for param,val in best_params.items():
        print(param,val)
    model = grid_search.best_estimator_
    model.fit(train_x,train_y)
    return model


# 目标划分
def trainlabel(x,label,test_size):
    X_train, X_test, y_train, y_test = train_test_split(x, label, test_size=test_size, random_state=0)
    return X_train,X_test,y_train,y_test


def main():
    # 获取数据
    xs, label_stabs, label_repus, label_acts, label_commus, label_resos, label_abilitys = dataprepare()
    labels = [label_stabs, label_repus,label_acts, label_commus, label_resos, label_abilitys]
    for i in range(len(labels)):
        print('第{}个标签\n'.format(i+1))
        X_train, X_test, y_train, y_test = trainlabel(xs, labels[i], test_size=0.1)
        # 线性回归
        clf = LinearRegression()
        model = clf.fit(X_train, y_train)
        predict = model.predict(X_test)
        print(mean_squared_error(y_test, predict))
        print(mean_absolute_error(y_test, predict))
        # GBRT
        clf = GradientBoostingRegressor(loss='ls',subsample=0.85,max_depth=5,n_estimators = 30)
        learning_rate = [0.01,0.05,0.08,0.1,0.2]
        max_depth = range(20,100,20)
        n_estimators = range(20,100,20)
        tuned_parameters =dict(learning_rate = learning_rate,max_depth = max_depth,n_estimators = n_estimators)
        model = gridsearch(X_train,y_train,clf,tuned_parameters,scoring = 'neg_mean_squared_error')
        predict = model.predict(X_test)
        print(mean_squared_error(predict,y_test))
        print(mean_absolute_error(predict,y_test))
        # RandomForest
        clf = RandomForestRegressor()
        max_depth = range(20,100,20)
        n_estimators = range(20,100,20)
        tuned_parameters =dict(max_depth = max_depth,n_estimators = n_estimators)
        model = gridsearch(X_train,y_train,clf,tuned_parameters,scoring = 'neg_mean_squared_error')
        predict = model.predict(X_test)
        print(mean_squared_error(predict, y_test))
        print(mean_absolute_error(predict, y_test))



if __name__ == '__main__':
    main()