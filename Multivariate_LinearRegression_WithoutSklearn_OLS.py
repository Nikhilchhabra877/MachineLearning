##### Developed by Nikhil ########
import  pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score

class data_processing:

    def fetch_data(self):
        X,y=load_diabetes(return_X_y=True)
        return X,y

    def train_test_split_(self,X,y):

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        return X_train,X_test,y_train,y_test

class MultivariateLinearRegression(data_processing):
    def __init__(self):
        self.coef = None
        self.intercept = None

    def fit(self,X_train,y_train):
        X_train = np.insert(X_train,0,1,axis=1)
        #return X_train.shape,y_train.shape

        ##Coeffs.

        m_values = np.linalg.inv(np.dot(X_train.T,X_train)).dot(X_train.T).dot(y_train)
        self.coef=m_values[1:]
        self.intercept=m_values[0]
        return self.coef,self.intercept

    def predict(self,X_test):
        y_pred=np.dot(X_test,self.coef) + self.intercept
        R2Score = r2_score(y_test,y_pred)
        return "R2-Score:" ,np.round(R2Score*100,2)

dp = data_processing()
X,y  = dp.fetch_data()
X_train,X_test,y_train,y_test = dp.train_test_split_(X,y)
lr = MultivariateLinearRegression()
print(lr.fit(X_train,y_train))
print(lr.predict(X_test))