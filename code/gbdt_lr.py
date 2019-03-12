# -*- ecoding:utf-8 -*-
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

# import tensorflow as tf

train_data = pd.read_table('../data/oppo_round1_train_20180929.txt',
        names= ['prefix','query_prediction','title','tag','label'], header= None, encoding='utf-8').astype(str)
val_data = pd.read_table('../data/oppo_round1_vali_20180929.txt',
        names = ['prefix','query_prediction','title','tag','label'], header = None, encoding='utf-8').astype(str)
test_data = pd.read_table('../data/oppo_round1_test_A_20180929.txt',
        names = ['prefix','query_prediction','title','tag'],header = None, encoding='utf-8').astype(str)
train_data.drop_duplicates(subset=['prefix','query_prediction','title','tag','label'], keep='first')
val_data.drop_duplicates(subset=['prefix','query_prediction','title','tag','label'], keep='first')
train_data = train_data[train_data['label'] != '音乐']
test_data['label'] = -1

train_data = pd.concat([train_data,val_data])
train_data['label'] = train_data['label'].apply(lambda x: int(x))
test_data['label'] = test_data['label'].apply(lambda x: int(x))
items = ['prefix', 'title', 'tag']

for item in items:
    temp = train_data.groupby(item, as_index = False)['label'].agg({item+'_click':'sum', item+'_count':'count'})
    temp[item+'_ctr'] = temp[item+'_click']/(temp[item+'_count'])
    train_data = pd.merge(train_data, temp, on=item, how='left')
    test_data = pd.merge(test_data, temp, on=item, how='left')
for i in range(len(items)):
    for j in range(i+1, len(items)):
        item_g = [items[i], items[j]]
        temp = train_data.groupby(item_g, as_index=False)['label'].agg({'_'.join(item_g)+'_click': 'sum','_'.join(item_g)+'count':'count'})
        temp['_'.join(item_g)+'_ctr'] = temp['_'.join(item_g)+'_click']/(temp['_'.join(item_g)+'count']+3)
        train_data = pd.merge(train_data, temp, on=item_g, how='left')
        test_data = pd.merge(test_data, temp, on=item_g, how='left')
train_data_ = train_data.drop(['prefix', 'query_prediction', 'title', 'tag'], axis = 1)
test_data_ = test_data.drop(['prefix', 'query_prediction', 'title', 'tag'], axis = 1)

print('train beginning')

X = np.array(train_data_.drop(['label'], axis = 1))
y = np.array(train_data_['label'])
X_test_ = np.array(test_data_.drop(['label'], axis = 1))
print('================================')
print(X.shape)
print(y.shape)
print('================================')

# 划分lgb训练集和lr特征训练集
X_train, X_train_lr, y_train, y_train_lr= train_test_split(X,y,test_size= 0.5)
X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,test_size=0.3)

grd = GradientBoostingClassifier(n_estimators= 10)
tr_enc = OneHotEncoder(categories='auto')
tr_enc.fit(grd.apply(X_train)[:,:,0])
X_train_enc = tr_enc.transform(grd.apply(X_train_lr)[:,:,0])
X_test_enc = tr_enc.transform(grd.apply(X_test_))

gbm_lr = LogisticRegression(solver='lbfgs', max_iter=1000)
gbm_lr.fit(X_train_enc,y_train_lr)
y_pred_gbm_lr = gbm_lr.predict(X_test_enc)[:,1]
print(f1_score(y_test,y_pred_gbm_lr))

