from logistic回归 import logistics_regression
from logistic回归 import Wite_and_load
import numpy as np

if __name__ == '__main__':
    print("******1.load the data**********")
    feature, label = Wite_and_load.load_data("data.txt")
    print("******2.train the model********")
    w = logistics_regression.cost_fun(feature, label, 1000, 0.1)
    print("The weight is" + str(w.T))
    print("******3.save the model*********")
    Wite_and_load.save_mode("model.txt",w)
    print("******4.test the model*********")
    w = Wite_and_load.load_weight("model.txt")
    test_feature = Wite_and_load.load_test_data("test_data")
    logistics_regression.predict(w, test_feature, "result.txt")
