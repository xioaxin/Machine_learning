# encoding :utf-8
import numpy as np



def load_data(filename):  # load the data
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


def write_data(filename):
    f = open(filename, 'w')
    feature = []
    label = []
    for line in f.readlines():
        feature_temp = []
        label_temp = []
        temple = line.split(" ")
        feature_temp.append(1)  # add the bias term
        for i in range(len(temple) - 1):
            feature_temp.append(float(temple[i]))
        label_temp.append(float(temple(-1)))
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
