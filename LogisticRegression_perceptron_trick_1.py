### Developed by Nikhil #############

'''

-Logistic Regression:

Perceptron trick is good and easily way to classify the classes. However, it does't give the best result we get from
the sklearn Logistic Regression class that uses sigmoid function internally to classify the data.

We will implement the same in part-2.

'''
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


class perceptron:

    def __init__(self,epochs,lr):
        self.epochs = epochs
        self.lr = lr

    def binary_con(self,value):
        return 1 if value > 0 else 0

    def fit(self,X,y):
        X = np.insert(X, 0, 1, axis=1)
        coefs_ = np.ones(X.shape[1])
        for i in range(self.epochs):
            id = np.random.randint(1,X.shape[0])
            y_pred = self.binary_con(np.dot(X[id],coefs_))
            coefs_ = coefs_ + self.lr*(y[id] - y_pred)*X[id]

        return coefs_[0],coefs_[1:]

    def plot_boundary(self,slope,intercept,X,y):

        m = -(slope[0] / slope[1])
        b = -(intercept / slope[1])
        X_data = np.linspace(-3,3,100)
        y_data = m * X_data + b
        plt.figure(figsize=(10, 6))
        plt.plot(X_data,y_data, color='red', linewidth=3)
        plt.scatter(X[:,0], X[:,1], c=y, cmap='winter', s=100)
        plt.ylim(-3, 2)
        plt.show()

    def sklerarn_LR(self,X,y):

        lr = LogisticRegression()
        lr.fit(X,y)
        coef = lr.coef_
        intercept = lr.intercept_
        m = -(coef[0][0]/coef[0][1])
        b = -(intercept/coef[0][1])
        return m,b


data = data_()
X,y=data.fetch_data()
print(X.shape)
print(y.shape)
#data.plot_data(X,y)

lg = perceptron(1500,0.01)
intercept,coef = lg.fit(X,y)
print(intercept)
print(coef)

lg.plot_boundary(coef,intercept,X,y)
m = -(coef[0]/coef[1])
b = -(intercept/coef[1])
print("Slope:",m)
print("Intercept",b)

print(lg.sklerarn_LR(X,y))

