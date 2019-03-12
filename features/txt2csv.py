# -*- ecoding:utf-8 -*-

"""
brief: 转换文件使用，注意更改文件path
author： Zhanglei
"""

import csv

f = '../data/test.txt'
with open('../data/test_data.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, dialect='excel')
    with open(f, 'rb') as filein:
        for line in filein:
            line_list = line.strip('\n').split('\t')
            spamwriter.writerow(line_list)

p = '../data/train.txt'
with open('../data/train_data.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, dialect='excel')
    with open(p, 'rb') as filein:
        for line in filein:
            line_list = line.strip('\n').split('\t')
            spamwriter.writerow(line_list)