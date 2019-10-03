# encoding :utf-8
import numpy as np
import math

'''
logistics regression
1.def the cost function sigmoid
2.use the  decrease  gradient
3.calculate the error rate
'''


def sig(x):
    '''
    :param x: input the W*X + b as x
    :return: the result between 0 and 1 ,indicate the rate to class
    '''
    return 1.0 / (1 + np.exp(-x))


def err_rate(h, label):
    '''
    :param h:   the value of sigmoid
    :param label:
    :return:   the error rate
    '''
    n = np.shape(h)[0]
    err_rate = 0.0
    for i in range(n):
        if h[i, 0] > 0 and (1 - h[i, 0]) > 0:
            err_rate -= label[i, 0] * np.log(h[i, 0]) + (1 - label[i, 0]) * np.log(1 - h[i, 0])
        else:
            err_rate -= 0
    return err_rate / n


def cost_fun(feature, label, max_cycle=1000, alpha=0.01):
    '''
    :param feature:
    :param label:  the label of the simple
    :param max_cycle: the max cycle times of simple
    :param alpha: the rate of learning
    :return:    the weight of simple
    '''
    n = np.shape(feature)[1]  # get the dimension of feature
    w = np.mat(np.ones((n, 1)))  # init the W
    i = 0
    while i <= max_cycle:
        i += 1
        h = sig(feature * w)  # calculate the value of sigmoid
        # print(h)
        err = label - h  # calculate the error
        if i % 100 == 0:
            print("the error rate of number " + str(i) + " is " + str(err_rate(h, label)))
        w = w + alpha * feature.T * err  # update the weight  (using the decrease gradient)
    return w


def predict(W, feature, filename):
    '''
    predict the test_data and get the class result
    :param W: the weight after training
    :param feature: the test feature
    :param filename: the name of save result file
    :return:
    '''
    n = np.shape(feature)[0]
    f = open(filename, "w")
    for i in range(n):
        result = np.dot(feature[i], W)
        if result > 0.5:
            f.writelines(str(1) + "\n")
        else:
            f.writelines(str(0) + "\n")
    f.close()
