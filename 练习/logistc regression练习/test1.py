# encoding:utf-8
import numpy as np


def sigmoid(x):
    '''
    sigmoid函数
    :param x:
    :return:
    '''
    return 1.0 / (1 + np.exp(-x))


def load_data(filename):
    '''
    读取文件的内容
    :param filename:
    :return: feature(特征),label(标签)
    '''
    f = open(filename, 'r')
    lines = f.readlines()
    feature = []
    label = []
    for line in lines:
        temple_feature = []
        temple_label = []
        temple = line.split("\t")
        temple_feature.append(1)  # 添加偏置项
        for i in range(len(temple) - 1):
            temple_feature.append(float(temple[i]))
        temple_label.append(float(temple[-1]))
        feature.append(temple_feature)
        label.append(temple_label)
    return np.mat(feature), np.mat(label)


def save_mode(filename, w):
    '''
    将权重写入到指定的文件
    :param filename:
    :param w:
    :return:
    '''
    f = open(filename, 'w')
    temple = []
    for i in range(len(w)):
        temple.append(str(w[i, 0]))
    f.write("\t".join(temple))
    f.close()


def cost(h, label):
    m = np.shape(h)[0]
    sum_err = 0
    for i in range(m):
        if h[i, 0] > 0 and 1 - h[i, 0] > 0:
            sum_err -= (label[i, 0] * np.log(h[i, 0]) + (1 - label[i, 0]) * np.log((1 - h[i, 0])))
        else:
            sum_err -= 0
    return sum_err / m


def gradient_logist(feature, label, max_Cycle, alpha):
    '''
    采用梯度下降法进行训练
    :param feature:
    :param label:
    :param max_Cycle:
    :param alpha:
    :return:
    '''
    n, m = np.shape(feature)
    w = np.ones((m, 1))  # 初始化权重，全部为1
    i = 0
    for i in range(max_Cycle):
        i += 1
        h = sigmoid(feature * w)
        error = label - h
        if i % 100 == 0:
            print('The error rate of the number ' + str(i) + " is " + str(cost(h, label)))
        w = w + alpha * feature.T * error  # 利用梯度下降法更新权重
    return w


def load_weight(filename):
    '''
    加载指定名称的权重
    :param filename:
    :return:
    '''
    f = open(filename, 'r')
    line = f.readline()
    temple = line.split("\t")
    weight = []
    for i in range(len(temple)):
        weight.append(float(temple[i]))
    return weight
    f.close


def load_test_data(filename):
    '''
    加载测试数据
    :param filename:
    :return:
    '''
    f = open(filename, 'r')
    feature = []
    lines = f.readlines()
    for line in lines:
        temple = line.split("\t")
        temple_feature = []
        temple_feature.append(1)
        for i in range(len(temple)):
            temple_feature.append(float(temple[i]))
        feature.append(temple_feature)
    return feature


def predicet(feature, filename, w):
    '''
    进行预测并将结果保存到指定的文件之中
    :param feature:
    :param filename:
    :param w:
    :return:
    '''
    label = feature * w
    f = open(filename, "w")
    for i in range(len(label)):
        if label[i, 0] >= 0:
            f.write(str(1) + "\n")
        else:
            f.write(str(0) + "\n")
    f.close()


if __name__ == '__main__':
    feature, label = load_data('data.txt')
    print(feature)
    w = gradient_logist(feature, label, 5000, 0.01)
    print(w)
    save_mode('model.txt', w)
    feature = load_test_data("test_data")
    print(feature)
    predicet(feature, 'result.txt', w)
