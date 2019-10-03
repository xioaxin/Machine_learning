import numpy as np
from  softmax回归 import softmaxRegres

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
        label_temple = []
        temple = line.split("\t")
        feature_temple.append(1)  # add the basis item
        for i in range(len(line) - 1):
            feature_temple.append(line[i])
        label_temple.append(line[-1])
        feature.append(feature_temple)
        label.append(label_temple)
    f.close()
    return np.mat(feature), np.mat(label)
def save_model(filename,w):
    f=open(filename,'w')
    n=np.shape(w)[0]
    for i in range(n):
        f.write(w[i,0]+"\t")
    f.close()



if __name__ == '__main__':
    # 1.load the data
    feature, label = load_data("Softinput.txt")
    # 2.train
    w=softmaxRegres.gradient_Ascent(feature,label,)
