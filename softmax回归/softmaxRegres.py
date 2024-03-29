# encoding :utf-8
import numpy as np

'''
soft_max_Regression 
'''


def cost(err, label):
    '''
    cost function
    :param err:
    :param label:
    :return:
    '''
    m = np.shape(err)[0]
    sum_cost = 0.0
    for i in range(m):
        if err[i, label[i, 0]] / np.sum(err[i, :]) > 0:
            sum_cost -= np.log(err[i, label[i, 0]] / np.sum(err[i, :]))
        else:
            sum_cost -= 0
    return sum_cost / m


def gradient_Ascent(feature, label, k, max_Cycle, alpha):
    '''
    using the gradient to class
    :param feature:
    :param label:
    :param l:
    :param max_Cycle:
    :param alpha:
    :return: w  (the wight of model)
    '''
    n, m = np.shape(feature)  # get the dimension of the feature
    weight = np.mat(np.ones((m, k)))  # init the weight
    # print(feature)
    i = 0  # iter sign
    while i <= max_Cycle:
        i += 1
        err = (feature * weight)  # calculate the result of the feature time Weight
        if i % 100 == 0:
            print("the cost of the number " + str(i) + " times is  " + str(cost(err, label)))
        row_sum = -err.sum(axis=1)  # calculate the sum of the error ang make the result be the  denominator
        row_sum = row_sum.repeat(k, axis=1)
        err = err / row_sum
        for x in range(n):
            err[x, label[x, 0]] += 1
        weight = weight + (alpha / n) * feature.T * err  # using gradient algorithm
    return weight
