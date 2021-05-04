import numpy as np

class LinearRegression:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.theta = None

    def fit(self):
        X = self.x
        Y = self.y
        # self.theta = np.linalg.inv(X.dot(np.transpose(X))).dot(np.transpose(X)).dot(Y))
        self.theta = np.linalg.inv(np.transpose(X).dot(X)).dot(np.transpose(X)).dot(Y)

    def predict(self,x):
        self.y_pred = x*self.theta

    # def evaluate(self):