# -*- coding: utf-8 -*-
"""
Created on Thu May 10 15:56:34 2018

@author: AI.Wong

E-mail:wanglei_edu@126.com
"""

'''
要求：电影评分男女的差异，即先得到每个人的平均分，再分男女得到各自的std
'''

import pandas as pd

# read u.user，得到users with unames columns
unames = ['user id','age','gender','occupation','zip code']
users = pd.read_table('ml-100k\\u.user',sep='|', names=unames)
### 内容显示提示，users应该用seq='|'，而ratings是seq='\t'
# read u.data, 得到ratings
rnames = ['user id','item id','rating','timestamp']
ratings = pd.read_table('ml-100k\\u.data',sep='\t',names=rnames)

# 合并数据,合并一依据就是相同的名称user id
mergeData = pd.merge(users,ratings, on='user id')
#mergeData = pd.merge(users,ratings, how='inner')

#以下是个很简洁的办法
'''
# mean Male
MeanDataM = mergeData[mergeData.gender=='M'].groupby('user id').rating.mean().std()
# mean Female
MeanDataF = mergeData[mergeData.gender=='F'].groupby('user id').rating.mean().std()
'''

'''
另外一种从透视表获得
先得到按id的平均   如果meanDf = mergeData.groupby('user id').rating.mean()也不可行
用透视表pivot_table再按M和F来求std
'''

pivotData = pd.pivot_table(mergeData,index='user id',values='rating',columns='gender',aggfunc='mean')
meanM = pivotData['M'].dropna(axis=0).std()
meanF = pivotData['F'].dropna(axis=0).std()
#print(meanM,meanF)
with open('movie_assgnment.txt','w') as f:
    f.write("{:.0f}{:.0f}".format(meanM*100, meanF*100))
