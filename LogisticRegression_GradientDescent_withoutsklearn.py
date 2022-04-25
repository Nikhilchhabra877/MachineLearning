###### Developed by : Nikhil #########
import numpy as np
from sklearn.datasets import  make_classification
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

class data_:

    def fetch_data(self):
        X, y = make_classification(n_samples=100, n_features=2, n_informative=1,n_redundant=0,
                           n_classes=2, n_clusters_per_class=1, random_state=41,hypercube=False,class_sep=10)
        return X, y

    def plot_data(self,X,y):
        plt.figure(figsize=(10, 6))
        plt.scatter(X[:,0],X[:,1],c=y)
        plt.show()

    def check_line(self):
        pass


class LogisticRegressionGD:

    def __init__(self,epochs,learning_rate):
        self.epochs=epochs
        self.learning_rate=learning_rate
    def sigmod(self,x):
        return 1/(1+np.exp(-x))

    def fit(self,X,y):
        X = np.insert(X,0,1,axis=1)
        weights = np.ones(X.shape[1])

        for i in range(self.epochs):
            y_hat =self.sigmod(np.dot(X,weights))
            weights = weights + self.learning_rate*(np.dot((y-y_hat),X)/X.shape[0])
        return weights[1:],weights[0]

dp = data_()
X,y=dp.fetch_data()
lr=LogisticRegressionGD(6500,0.5)
print(lr.fit(X,y))
lr_sk = LogisticRegression(penalty='none',solver='sag')
lr_sk.fit(X,y)
print()
print(lr_sk.coef_)
print(lr_sk.intercept_)
