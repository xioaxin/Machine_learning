# encoding :utf-8
import numpy as np


def load_data(filename):
    """
    load the original data to training
    :param filename:
    :return:
    """
    f = open(filename, 'r')
    feature = []
    label = []
    for line in f.readlines():
        feature_temp = []
        label_temp = []
        temple = line.split("\t")  # split by "\t "
        feature_temp.append(float(1))  # add the bias term
        for i in range(len(temple) - 1):
            feature_temp.append(float(temple[i]))
        label_temp.append(float(temple[-1]))
        feature.append(feature_temp)
        label.append(label_temp)
    f.close()
    return np.mat(feature), np.mat(label)


def save_mode(filename, W):  # save model in the file model.txt
    f = open(filename, 'w')
    n = np.shape(W)[0]
    content = []
    for i in range(n):
        content.append(str(W[i, 0]))
    r = f.write("\t".join(content))
    f.close()


def load_weight(filename):
    '''
    get the weight of model from the file
    :param filename:
    :return:
    '''
    f = open(filename, 'r')
    temp = f.readline().split("\t")
    w = np.ones((3,1))
    # print(w)
    for i in range(len(temp)):
        w[i][0] = float(temp[i])
    f.close()
    return w


def load_test_data(filename):
    '''
    laod the test data to predict the class result
    :param filename:
    :return:
    '''
    f = open(filename, 'r')
    lines = f.readlines()
    test_feature = []
    for var in lines:
        line = var.split("\t")
        temp_feature = []
        temp_feature.append(1)
        for i in range(len(line)):
            temp_feature.append(float(line[i]))
        test_feature.append(temp_feature)
    f.close()
    return test_feature
