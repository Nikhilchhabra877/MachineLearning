### Developed by : Nikhil ###################
## Ridge Regression Mathamatical Intution and implementation####

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression

class data:

    def fetch_data(self):
        X,y=load_diabetes(return_X_y=True)
        return X,y

    def train_test_split_(self,X,y):

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        return X_train,X_test,y_train,y_test
    def random_data(self):
        X, y = make_regression(n_samples=100, n_features=1, n_informative=1, n_targets=1, noise=20, random_state=13)
        return X,y

class Ridge_LinearRegression():

    def __init__(self,alpha=0.1):
        self.alpha= alpha
        self.coef_ = None
        self.intercept = None

    def fit(self,X_train,y_train):

        num = 0
        den = 0

        for i in range(X_train.shape[0]):

            num = num + (y_train[i] - y_train.mean()) * (X_train[i] - X_train.mean())
            den = den + (X_train[i] - X_train.mean()) * (X_train[i] - X_train.mean())

        self.coef_ = num/(den + self.alpha)
        self.intercept = y_train.mean() - (self.coef_*X_train.mean())
        return "Without using sklearn:", self.coef_,self.intercept

class MultivarRidge:
    def __init__(self,alpha=0.01):
        self.alpha = alpha
        self.coef_ = None
        self.intercept = None

    def fit(self,X_train,y_train):
        X_train = np.insert(X_train,0,1,axis=1)
        I = np.identity(X_train.shape[1])
        I[0][0] = 0
        result = np.linalg.inv(np.dot(X_train.T,X_train) + self.alpha * I).dot(X_train.T).dot(y_train)
        self.intercept = result[0]
        self.coef_ = result[1:]
        return self.intercept, self.coef_

    def predict(self,X_test):
        return np.dot(X_test,self.coef_) + self.intercept

rr = Ridge_LinearRegression(alpha=100)
dp= data()
X,y=dp.random_data()
#print(rr.fit(X,y))
rr=Ridge(alpha=100)
rr.fit(X,y)
#print("Using sklearn:", rr.coef_,rr.intercept_)

X,y = dp.fetch_data()
X_train,X_test,y_train,y_test = dp.train_test_split_(X,y)
rr_mul = Ridge(alpha=0.1,solver='cholesky')
rr_mul.fit(X_train,y_train)
print(rr_mul.coef_)
print(rr_mul.intercept_)
print("With Sklearn:",r2_score(y_test,rr_mul.predict(X_test))*100)

print()

rr_custom = MultivarRidge(alpha=0.1)
print(rr_custom.fit(X_train,y_train))
print("Without Sklearn:",r2_score(y_test,rr_custom.predict(X_test))*100)


## We can see both the classes(SKLearn and Custom ridge) are giving same results.