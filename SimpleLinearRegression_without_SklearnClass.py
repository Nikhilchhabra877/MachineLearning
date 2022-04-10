### Developed by : Nikhil ##########

import pandas as pd
from sklearn.model_selection import train_test_split

class data_processing:

    def __init__(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train

    def fetch_data(self,path):
        data = pd.read_csv(path)
        return data


    def train_test_split_(self,data):

        X=data.iloc[:,0].values
        y=data.iloc[:,1].values
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        return self.X_train,self.y_train,self.X_test,self.y_test

class SimpleLinearRegression(data_processing):

    def __init__(self):
        self.slope=None
        self.intercept=None


    def fit(self,X_train,y_train):
        n = 0
        d = 0
        for i in range(X_train.shape[0]):
           n = n + ((X_train[i] - X_train.mean()) * (y_train[i] - y_train.mean()))
           d = d + ((X_train[i] - X_train.mean()) * (X_train[i] - X_train.mean()))

        self.slope= n/d
        self.intercept = y_train.mean() - (self.slope * X_train.mean())
        return self.slope,self.intercept

    def predict(self,X_test):
        return self.slope * X_test + self.intercept

lr = SimpleLinearRegression()
data = lr.fetch_data("/Users/nikhil/Desktop/placement.csv")
trf_data = lr.train_test_split_(data)
print(lr.fit(trf_data[0],trf_data[1]))
print(lr.predict(6.89))