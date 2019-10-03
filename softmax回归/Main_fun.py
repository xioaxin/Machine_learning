# encoding :utf-8

import numpy as np
from softmax回归 import softmaxRegres
from softmax回归 import predict_fun


def load_data(filename):
    '''
    load the train_data
    :param filename:
    :return: feature ,label
    '''
    f = open(filename, 'r')
    lines = f.readlines()
    feature = []
    label = []
    for line in lines:
        feature_temple = []
        temple = line.split("\t")
        feature_temple.append(1)  # add the basis item
        for i in range(len(temple) - 1):
            feature_temple.append(float(temple[i]))
        label_temple = (int(temple[-1]))
        feature.append(feature_temple)
        label.append(label_temple)
    f.close()
    return np.mat(feature), np.mat(label).T, len(set(label))


def save_model(filename, w):
    '''
    save the weight of the model in the defined file
    :param filename:
    :param w:
    :return:
    '''
    f = open(filename, 'w')
    n, m = np.shape(w)
    for i in range(n):
        for j in range(m):
            f.write(str(w[i, j]) + "\t")
        f.write("\n")
    f.close()


if __name__ == '__main__':
    # 1.load the data
    print('******1.load data********************')
    feature, label, k = load_data("Softinput.txt")
    # 2.train
    print('*****2.train data********************')
    w = softmaxRegres.gradient_Ascent(feature, label, k, 5000, 0.2)
    print(w)
    # # 3.save the model
    print('*****3.save model********************')
    save_model("result.txt", w)
    print('*****4.load test_data****************')
    w = predict_fun.load_weight("result.txt")
    data = predict_fun.load_data(4000, 3)
    print('*****5.predict test_data*************')
    predict_fun.predict(data,w,'predict_resule.txt')
