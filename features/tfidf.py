# -*- ecoding:utf-8 -*-

"""
transform tfidf特征
author: Zhanglei
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import time

time_start = time.time()

"""=====================================================================================================================
1 数据预处理
"""
df_train = pd.read_csv('../data/train_data.csv') # 测试用，需要改文件路径
df_test = pd.read_csv('../data/test_data.csv')
#print(df_train)
#print(df_test)
df_all = df_train.append(df_test)
#print(df_all)
y_train = (df_train['label']).values

"""=====================================================================================================================
2 特征工程

"""
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True)
vectorizer.fit(df_all)
x_train = vectorizer.transform(df_train)
x_test = vectorizer.transform(df_test)

"""=====================================================================================================================
3 保存至本地
"""
data = (x_train, y_train, x_test)
fp = open('./data_tfidf.pkl', 'wb')
pickle.dump(data, fp)
fp.close()

t_end = time.time()
print("已将原始数据数字化为tf特征，共耗时：{}min".format((t_end-t_start)/60))


