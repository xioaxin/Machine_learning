from logistic回归 import logistics_regression
from logistic回归 import Wite_and_load
import  numpy as np
if __name__ == '__main__':
    print("******1.load the data**********")
    feature, label = Wite_and_load.load_data("data.txt")
    w=logistics_regression.cost_fun(feature, label, 1000, 0.1)
    print("权重为: "+str(w.T))
    print()
    Wite_and_load.save_mode("model.txt",w)

