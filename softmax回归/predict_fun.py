# encoding :utf-8
import numpy as np
import random as rd

def load_weight(filename):
    f = open(filename, "r")
    lines = f.readlines()
    weight = []
    for line in lines:
        temple = []
        weight_temple = []
        temple = line.split("\t")
        for i in range(len(temple)):
            if temple[i] != '\n':
                weight_temple.append(float(temple[i].lstrip('\n')))
        weight.append(weight_temple)
    return weight


def load_data(num, m):
    '''导入测试数据
    input:  num(int)生成的测试样本的个数
            m(int)样本的维数
    output: testDataSet(mat)生成测试样本
    '''
    testDataSet = np.mat(np.ones((num, m)))
    for i in range(num):
        testDataSet[i, 1] = rd.random() * 6 - 3  # 随机生成[-3,3]之间的随机数
        testDataSet[i, 2] = rd.random() * 15  # 随机生成[0,15]之间是的随机数
    return testDataSet


def predict(data, w, filename):
    n, m = np.shape(data)
    print(data)
    print(w)
    h = data * w
    result = h.argmax(axis=1)
    f = open(filename, 'w')
    n = np.shape(result)[0]
    for i in range(n):
        f.write(str(result[i, 0]) + '\n')
    f.close()
