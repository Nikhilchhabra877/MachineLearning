######## Developed by : Nikhil #################
## Ridge regression using Graident Descent - Mathamatical Implementation ######

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression
from sklearn.linear_model import SGDRegressor



class data:

    def fetch_data(self):
        X,y=load_diabetes(return_X_y=True)
        return X,y

    def train_test_split_(self,X,y):

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        return X_train,X_test,y_train,y_test

class Ridge_GradientDescent:

    def __init__(self, epochs, learning_rate, alpha):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None

    def fit(self,X_train,y_train):

        self.coef_ = np.ones(X_train.shape[1])
        self.intercept_ = 0
        weights = np.insert(self.coef_, 0, self.intercept_)
        X_train = np.insert(X_train,0,1,axis=1)

        for i in range(self.epochs):

            weights_der = np.dot(X_train.T,X_train).dot(weights) - np.dot(X_train.T,y_train) + self.alpha* weights
            weights = weights - self.learning_rate * weights_der

        self.coef_ = weights[1:]
        self.intercept_ = weights[0]

        #return self.coef_, self.intercept_

    def predict(self, X_test):
        return np.dot(X_test, self.coef_) + self.intercept_

dp=data()
X,y = dp.fetch_data()
X_train,X_test,y_train,y_test = dp.train_test_split_(X,y)
rgd = Ridge_GradientDescent(epochs=200,alpha=0.001,learning_rate=0.005)
rgd.fit(X_train,y_train)
print("R2_score without sklearn:",r2_score(y_test,rgd.predict(X_test)))

sgd = SGDRegressor(penalty='l2',max_iter=200,learning_rate='constant',alpha=0.001)
sgd.fit(X_train,y_train)
print("R2_score using sklearn SGD Regressor:",r2_score(y_test,sgd.predict(X_test)))


