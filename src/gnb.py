import numpy as np
import math

class gnb:
    __slots__ = ['prob', 'std', 'mean']

    def __init__(self):
        self.prob = [0, 0]
        self.std = []
        self.mean = []

    def train(self, x, y):
        self.std = np.zeros((2, x.shape[1]))
        self.mean = np.zeros((2, x.shape[1]))

        for i in range(2):
            self.prob[i] = self.county(y, i)
            index_db = y[y == i].index
            for j in range(len(x.columns)):
                column = x.columns[j]
                self.std[i][j] = np.std(x[column].loc[index_db])
                self.mean[i][j] = np.mean(x[column].loc[index_db])

    def county(self, y, yp):
        c = 0
        for i in range(y):
            if i==yp:
                c+=1
        return c

    def ydadox(self, x, y):
        resultado = 1
        for num_column in range(len(x)):
            std = self.std[y][num_column]
            mean = self.mean[y][num_column]
            resultado *= ((1 / (math.sqrt(2 * math.pi) * std)) * (
                math.exp((-(x[num_column] - mean) ** 2) / (2 * (std ** 2)))))
        return resultado

    def predict(self, x):
        total = []
        for y in range(len(self.prob)):
            total += [self.prob[y] * self.ydadox(x, y)]
        return int(np.argmax(total))

    def test(self, x, y):
        total = x.shape[0]
        acc = 0
        for i, j in zip(x, y):
            ypred = self.predict(i)
            if i == j:
                acc += 1

        return (acc / total)