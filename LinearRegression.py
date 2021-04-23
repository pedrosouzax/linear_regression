import numpy as np

class LinearRegression:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.theta = fit()

    def fit(self):
        X = self.x
        Y = self.y
        theta = np.linalg.inv(X.dot(np.transpose(X))).dot(np.transpose(X).dot(Y))
        return theta

    def evaluate():
        pass