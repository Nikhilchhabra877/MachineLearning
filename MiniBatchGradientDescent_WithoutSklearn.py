############## Developed by Nikhil #######################

'''
# This technique provides the good results as compared to stochastic GD  with minimum computations .
# We can use partial_fit function in the Stochastic GD to use this functionality. There is no special class in Sklearn
for this.
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
import random

class data_processing:

    def fetch_data(self):
        X,y=load_diabetes(return_X_y=True)
        return X,y

    def train_test_split_(self,X,y):

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        return X_train,X_test,y_train,y_test

class MBGradientDescent(data_processing):
    def __init__(self,epochs,batch_size,learning_rate):
        self.coef_ = None
        self.intercept_ = None
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def fit(self,X_train,y_train):
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1])

        for i in range(self.epochs):
            for j in range(int(X_train.shape[0]/self.batch_size)):
                batch = random.sample(range(X_train.shape[0]),self.batch_size)
                y_pred = np.dot(X_train[batch],self.coef_) + self.intercept_
                inter_der = -2 * np.mean(y_train[batch] - y_pred)
                self.intercept_ = self.intercept_ - (self.learning_rate * inter_der)

                coef_der = -2 * np.dot((y_train[batch] - y_pred),X_train[batch])
                self.coef_ = self.coef_ - (self.learning_rate * coef_der)

        return  self.intercept_,self.coef_
    def predict(self,X_test):
        return np.dot(X_test,self.coef_) + self.intercept_

dp=data_processing()
X,y=dp.fetch_data()
X_train,X_test,y_train,y_test = dp.train_test_split_(X,y)

mbd = MBGradientDescent(epochs=180,batch_size=int(X_train.shape[0]/50),learning_rate=0.01)
coef = mbd.fit(X_train,y_train)
y_pred = mbd.predict(X_test)
print(coef)
print("R2_score:", r2_score(y_test,y_pred))