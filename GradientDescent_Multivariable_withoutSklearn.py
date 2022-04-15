#### Developed by : Nikhil ###############
#### This included mathamatical intution and equations required to find coefs. in multivariate LR.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score

class data_processing:

    def fetch_data(self):
        X,y=load_diabetes(return_X_y=True)
        return X,y

    def train_test_split_(self,X,y):

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        return X_train,X_test,y_train,y_test

class GD_Regressor_mv(data_processing):

    def __init__(self,lr,epochs):
        self.coef_ = None
        self.intercept_ = None
        self.lr = lr
        self.epochs = epochs

    def fit(self,X_train,y_train):

        ###First initialize the coefs.
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1])

        ### Now, Calculate the value of intercept.
        for i in range(self.epochs):

            y_pred = np.dot(X_train,self.coef_) + self.intercept_
            b_intercept = -2 * np.mean(y_train - y_pred)
            self.intercept_ = self.intercept_ - (self.lr * b_intercept)

            ### Next, find the value of coefs.

            m_slope = -2 * np.dot((y_train-y_pred),X_train)/X_train.shape[0]
            self.coef_ = self.coef_ - (self.lr * m_slope)
        return self.intercept_,self.coef_

    def predict(self,X_test):
        y_pred = np.dot(X_test,self.coef_) + self.intercept_
        return y_pred

## Verify the details of training and test set.

dp = data_processing()

X,y = dp.fetch_data()
print(X.shape)
print(y.shape)

X_train,X_test,y_train,y_test = dp.train_test_split_(X,y)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

gd = GD_Regressor_mv(0.3,700)

print(gd.fit(X_train,y_train))

y_pred = gd.predict(X_test)
print("R2_Score :",r2_score(y_test,y_pred))

## Further, it can be improved and verified by changing the learning and epochs or verify the coef. with SKLEARN linear regression class.
