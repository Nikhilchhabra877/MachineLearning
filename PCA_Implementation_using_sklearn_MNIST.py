### Developed by  : Nikhil #####

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class data_processing:

    def fetch_data(self, path):
        data = pd.read_csv(path)
        return data

    def train_test_split_(self, data):
        X = data.iloc[:, 1:].values
        y = data.iloc[:, 0].values
        # self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        # splited_data = self.X_train,self.y_train,self.X_test,self.y_test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, y_train, X_test, y_test

    def scale(self, X_train, X_test):
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)
        return X_train, X_test

    def PCA_(self, X_train, X_test, n_features):
        pca = PCA(n_components=n_features)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.fit_transform(X_test)
        return X_train_pca, X_test_pca


class algorithm(data_processing):

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        # self.knn = knn

    def fit(self, X_train, X_test):
        global knn
        knn = KNeighborsClassifier()
        knn.fit(X_train, X_test)

    def predict(self, X_test):
        # knn = KNeighborsClassifier()
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return y_pred, "Accuracy score:", accuracy


dp = data_processing()
data = dp.fetch_data("/Users/nikhil/Downloads/train.csv")
X_train, y_train, X_test, y_test = dp.train_test_split_(data)
# X_train_trf,X_test_trf = dp.scale(X_train,X_test)
# X_train_pca,X_test_pca = dp.PCA_(X_train,X_test,500)

KNN = algorithm(X_train, X_test, y_train, y_test)
# KNN2 = algorithm(X_train_pca,X_test_pca,y_train,y_test)
KNN.fit(X_train, y_train)
print(KNN.predict(X_test))
# KNN2.fit(X_train_pca,y_train)
# KNN2.predict(X_test_pca)