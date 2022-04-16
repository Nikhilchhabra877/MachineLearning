### Developed by : Nikhil #######
'''
#####################################################################################################################
Problem with batch GD :
Suppose we have:
n(no. of rows): 1000
5 - Input(features)
6 - coef (b0 + b1*x1 + b2*x2 + b3*x3 + b4*x4 + b5*x5

# Np. of Calculations of for one coef. : 1 * 1000 and for 6 coefs. 1000 * 6 = 6000 and for 50 epochs = 6000 * 50 = 300000
derivatives(calculations)

- In Batch GD we calculate n derivatives in each epoch and take one step towards optimum solution.
- If we have dataset with higher dimensions it will be slower because of lots of computation.
- If the dataset is too large, we could not perform vectorizations if we have low memory.
- The reason is, we need to load the whole training data in the memory in each epoch for finding the chefs.

#####################################################################################################################
#### To solve this problem, we use Stochastic GD, where we take one random row from our dataset and move towards the
optimum solution.

# In this technique, in each epoch we take n no. of steps towards Global minima because of this it converges faster than
batch GD.
# This technique is widely used in deep learning where we work on images and text data.
# Pros:-
- faster convergence
- less computation
# Cons:
- Doesn't give steady solution because of randomness.

Also, Sometimes due to randomness , the result vary when we reach near to the global minima, to resolve this we use another
technique called as learning schedule, in this technique we change the learning rate after every epoch, in other words, we
make a function with respect to epoch. This technique helps us to converge faster and provides better solution.

'''

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

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
        return X_train,X_test,y_train,y_test

class SGDRegressor(data_processing):
    def __init__(self,epochs,learning_rate):
        self.coef_ = None
        self.intercept_ = None
        self.epochs = epochs
        self.lr = learning_rate

    def fit(self,X_train,y_train):
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1])

        for i in range(self.epochs):
            for j in range(X_train.shape[0]):

                ran_id = np.random.randint(0,X_train.shape[0])
                y_hat = np.dot(X_train[ran_id],self.coef_) + self.intercept_
                inter_der = -2 * (y_train[ran_id] - y_hat)
                self.intercept_ = self.intercept_ - (self.lr * inter_der)

                coef_der =  -2 * np.dot((y_train[ran_id] - y_hat),X_train[ran_id])
                self.coef_ = self.coef_ - (self.lr * coef_der)

        return self.intercept_,self.coef_

    def predict(self,X_test):
        return np.dot(X_test,self.coef_) + self.intercept_


dp = data_processing()
X,y = dp.fetch_data()
X_train,X_test,y_train,y_test = train_test_split(X,y)
print(X_train.shape[0])
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

sgd = SGDRegressor(40,0.01)
coef = sgd.fit(X_train,y_train)
y_pred = sgd.predict(X_test)
R2_score = r2_score(y_test,y_pred)
print(R2_score)
print(coef)