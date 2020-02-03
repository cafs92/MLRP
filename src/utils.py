import numpy as np
import pandas as pd
from sklearn import datasets


def readdata():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    y = changeclass(y)
    return x, y

def changeclass(y):
    for i in range(len(y)):
        if y[i] == 1:
            y[i] = 0
        else:
            y[i] = 1
    return y
