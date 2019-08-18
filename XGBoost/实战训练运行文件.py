# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 23:44:41 2019

@author: wanglei

开发环境：win10， python3.6
"""

import pandas as pd
import numpy as np
import datetime
import csv
import os
import scipy as sp
import xgboost as xgb
import itertools
import operator
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import TransformerMixin
from sklearn import cross_validation
from matplotlib import pylab as plt

plot = True

goal = 'Sales'
myid = 'Id'

def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w

# 自己定义的一些loss function
def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe
 
def rmspe_xg(yhat, y):
    # y = y.values
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))
    return "rmspe", rmspe

def load_data():
    """
        加载数据，设定数值型和非数值型数据
    """
    store = pd.read_csv('./data/store.csv')
    train_org = pd.read_csv('./data/train.csv',dtype={'StateHoliday':pd.np.string_})
    test_org = pd.read_csv('./data/test.csv',dtype={'StateHoliday':pd.np.string_})
    train = pd.merge(train_org,store, on='Store', how='left')
    test = pd.merge(test_org,store, on='Store', how='left')
    features = test.columns.tolist()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    features_numeric = test.select_dtypes(include=numerics).columns.tolist()
    features_non_numeric = [f for f in features if f not in features_numeric]
    return (train,test,features,features_non_numeric)

def process_data(train,test,features,features_non_numeric):
    """
        Feature engineering and selection.
    """
    # # FEATURE ENGINEERING
    train = train[train['Sales'] > 0]
 
    for data in [train,test]:
        # year month day
        data['year'] = data.Date.apply(lambda x: x.split('-')[0])
        data['year'] = data['year'].astype(float)
        data['month'] = data.Date.apply(lambda x: x.split('-')[1])
        data['month'] = data['month'].astype(float)
        data['day'] = data.Date.apply(lambda x: x.split('-')[2])
        data['day'] = data['day'].astype(float)
 
        # promo interval "Jan,Apr,Jul,Oct"
        data['promojan'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Jan" in x else 0)
        data['promofeb'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Feb" in x else 0)
        data['promomar'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Mar" in x else 0)
        data['promoapr'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Apr" in x else 0)
        data['promomay'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "May" in x else 0)
        data['promojun'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Jun" in x else 0)
        data['promojul'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Jul" in x else 0)
        data['promoaug'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Aug" in x else 0)
        data['promosep'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Sep" in x else 0)
        data['promooct'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Oct" in x else 0)
        data['promonov'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Nov" in x else 0)
        data['promodec'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Dec" in x else 0)
 
    # # Features set.
    noisy_features = [myid,'Date']
    features = [c for c in features if c not in noisy_features]
    features_non_numeric = [c for c in features_non_numeric if c not in noisy_features]
    features.extend(['year','month','day'])
    # Fill NA
    class DataFrameImputer(TransformerMixin):
        # http://stackoverflow.com/questions/25239958/impute-categorical-missing-values-in-scikit-learn
        def __init__(self):
            """Impute missing values.
            Columns of dtype object are imputed with the most frequent value
            in column.
            Columns of other types are imputed with mean of column.
            """
        def fit(self, X, y=None):
            self.fill = pd.Series([X[c].value_counts().index[0] # mode
                if X[c].dtype == np.dtype('O') else X[c].mean() for c in X], # mean
                index=X.columns)
            return self
        def transform(self, X, y=None):
            return X.fillna(self.fill)
    train = DataFrameImputer().fit_transform(train)
    test = DataFrameImputer().fit_transform(test)
    # Pre-processing non-numberic values
    le = LabelEncoder()
    #将label转化成0，1等
    for col in features_non_numeric:
        le.fit(list(train[col])+list(test[col]))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])
    # LR和神经网络这种模型都对输入数据的幅度极度敏感，请先做归一化操作
    scaler = StandardScaler()
    for col in set(features) - set(features_non_numeric) - set([]): # TODO: s[] used to add what not to scale
        scaler.fit( np.array(list(train[col]) + list(test[col])).reshape(-1,1) )     # 报错：新版本sklearn需要二位数组
        #可以直接使用训练集对测试集数据进行转换
        train[col] = scaler.transform( np.array(train[col]).reshape(-1,1))        #dataframe的一列是pd.Series，带索引的，不好乱转化
        test[col] = scaler.transform( np.array(test[col]).reshape(-1,1) )
    return (train,test,features,features_non_numeric)

def XGB_native(train, test, features, features_non_numeric):        #lightgbm更快一些，准确度是比xgb更高一些
    depth = 13
    eta = 0.01
    ntrees = 8000     #树太多可能over fiting
    mcw = 3
    params = {"objective": "reg:linear",    #这里参数不能多出空格
              "booster": "gbtree",
              "eta": eta,
              "max_depth": depth,
              "min_child_weight": mcw,
              "subsample": 0.9,
              "colsample_bytree": 0.7,
              "silent": 1
             }
    print("Running with params: " + str(params))
    print("Running with ntrees: " + str(ntrees))
    print("Running with feaures: " + str(features))
   
    # 以下没有做c-v，太花时间
    
    
    # Train model with Local split
    tsize = 0.05
    X_train, X_test = cross_validation.train_test_split(train, test_size = tsize)
    dtrain = xgb.DMatrix(X_train[features], np.log(X_train[goal] + 1))
    dvalid = xgb.DMatrix(X_test[features], np.log(X_test[goal] + 1))
    watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    
    #这里设置了early stopping, 100词之后loss没变则停止
    #loss function里面的编写，原版xgb需要返回grad一阶，二阶。不然有时候会报错。 这里的func写法没有返回grad，hess
    gbm = xgb.train(params, dtrain, ntrees, evals = watchlist, early_stopping_rounds = 100, feval = rmspe_xg, verbose_eval = True)
    train_probs = gbm.predict(xgb.DMatrix(X_test[features]))
    indices = train_probs < 0
    train_probs[indices] = 0
    error = rmspe(np.exp(train_probs) - 1, X_test[goal].values)
    print(error)
    
    # Predict and Export
    test_probs = gbm.predict(xgb.DMatrix(test[features]))
    indices = test_probs < 0
    test_probs[indices] = 0
    submission = pd.DataFrame({myid: test[myid], goal: np.exp(test_probs) - 1})
    if not os.path.exist('result/'):
        os.makedirs('result/')
    submission.to_csv("./result/data-xgb_d%s_eta%s_ntrees%s_mcw%s_tsize%s.csv" % (str(depth), str(eta), str(ntrees), str(mcw), str(tsize)))
    
    # Feature importance
    if plot:
        outfile = open('xgb.fmp', 'w')
        i = 0
        for feat in features:
            outfile.write('{0}\t{1}\tq\n'.format(i, feat))
            i = i + 1
        outfile.colse()
        importance = gbm.get_fscore(fmap = 'xgb.fmp')
        importance = sorted(importance.items(), key = operator.itemgetter(1))
        df = pd.DataFrame(importance, columns = ['feature', 'fscore'])
        df['fscore'] = df['fscore'] / df['fscore'].sum()
        # Plotitup
        plt.figure()
        df.plot()
        df.plot(kind = 'barh', x = 'feature', y = 'fscore', legend = False, figsize = (25, 15))
        plt.title('XGBoost Feature Importance')
        plt.xlable('relative importance')
        plt.gcf().savefig('Feature_Importance_xgbb_d%s_eta%s_ntrees%s_mcw%s_tsize%s.png' % (str(depth),str(eta),str(ntrees),str(mcw),str(tsize)))


print("=> 载入数据中...")
train,test,features,features_non_numeric = load_data()

print("=> 处理数据与特征工程...")
train,test,features,features_non_numeric = process_data(train,test,features,features_non_numeric)
print("=> 使用XGBoost建模...")
XGB_native(train,test,features,features_non_numeric)