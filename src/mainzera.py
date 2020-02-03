from sklearn.model_selection import train_test_split

import src.utils as ut
import src.gnb as nb
import numpy as np


def sharedmain(model, r):
    x, y = ut.readdata()
    acc = []
    for i in range(r):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)  # Holdout 80/20
        model.train(model, x_train, y_train)
        acc.append(model.test(x_test, y_test))
    return np.mean(acc)


gnb = nb.gnb
print(sharedmain(gnb, 20))