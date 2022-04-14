import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class data_:

    def fetch_data(self):
        X, y = make_regression(n_samples=100, n_features=1, n_informative=1, n_targets=1, noise=40)
        return X, y


class GDRegressor(data_):

    def __init__(self, epochs, learning_rate):
        self.m = 100
        self.b = -120
        self.epochs = epochs
        self.learning_rate = learning_rate

    def fit(self, X, y):
        for i in range(self.epochs):
            slope_b = -2 * np.sum(y - self.m * X.ravel() - self.b)
            slope_m = -2 * np.sum((y - self.m * X.ravel() - self.b) * X.ravel())

            self.b = self.b - (self.learning_rate * slope_b)
            self.m = self.m - (self.learning_rate * slope_m)
        return self.b, self.m

    def predict(self, X):
        y_pred = self.m * X.ravel() + self.b
        return y_pred

data = data_()
X,y=data.fetch_data()
gd= GDRegressor(150,0.001)
print(gd.fit(X,y))
print(gd.predict(X))
