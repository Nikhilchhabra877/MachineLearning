### By: Nikhil ###########
'''
Regularization refers to techniques that are used to calibrate machine learning models in order to minimize the adjusted
loss function and prevent overfitting or underfitting. Using Regularization, we can fit our machine learning model
appropriately on a given test set and hence reduce the errors in it.

- L1: Lasso
- L2: Ridge
- L3: Elastic net

- This prevents from overfitting. where ML model performed well in training but on test dataset it gives high errors.
- One trick to see that or model has overfitting problem or not is to see the value of m . if the value of m is too high
then it indicates that role of x (associated input with m) is very high in order to predict the output y and if it is
low then underfitting problem.

- In Ridge regression, we add one extra term in out loss function that will help to reduce the loss.

L = np.sum(np.sqrt(y_actual - y_pred)) + lm(m**m)

case-1 : let suppose, the line is passing through all the points. In this case our model is overfitting.

y=2.3,5.3
x = 1,3
m=1.5
b = 0.8
b2 =1.5

Since, the line is passing through all the data points in that case our (y_actual - y_pred) would be zero. So, the loss
would be : 0 + (1.5**1.5) = 2.25

case-2 :  when line is not passing through exactly like case-1.

equation : sqrt(2.3 - 1.5(1) - 1.5) + sqrt(5.3 - 0.9(3) - 1.5) + sqrt(0.9) = 2.03  --- loss reduced as compared to above
for same data points.

The whole idea is in this case bias increases little bit  but our model perform well on test data (variance decrease) after
applying this technique.

Like Linear regression, in this also we use GD to find best fit line.
'''

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.linear_model import  LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

class Ridge_regression:

    def fetch_data(self):
        X,y=load_diabetes(return_X_y=True)
        return X,y

    def train_test_split_(self,X,y):

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        return X_train,X_test,y_train,y_test
    def random_data(self):
        m = 100
        x1 = 5 * np.random.rand(m, 1) - 2
        x2 = 0.7 * x1 ** 2 - 2 * x1 + 3 + np.random.randn(m, 1)
        return x1,x2

    def plot(self,x1,x2):
        plt.scatter(x1, x2),plt.show()

    def ridge_randomdata(self,x1,x2,alpha):
        model = Pipeline([
            ('Ployfeatures',PolynomialFeatures(degree=200)),
            ('Ridge',Ridge(alpha=alpha))])
        model.fit(x1,x2)
        return model.predict(x1)


dp = Ridge_regression()
X,y = dp.fetch_data()
X_train,X_test,y_train,y_test = dp.train_test_split_(X,y)
X_random,y_random = dp.random_data()

lr = LinearRegression()
lr.fit(X_train,y_train)
print("R2_score before Regularization:",r2_score(y_test,lr.predict(X_test)))
print("RMSE before Regularization:",np.sqrt(mean_squared_error(y_test,lr.predict(X_test))))

print ("####################################################################################")

rr = Ridge(alpha=0.01)
rr.fit(X_train,y_train)
print("R2_score After Regularization:",r2_score(y_test,rr.predict(X_test)))
print("RMSE After Regularization:",np.sqrt(mean_squared_error(y_test,rr.predict(X_test))))

## It is performing slightly good as compared to linear regression because this model is not much complex. Further,we will
#test Lasso regression on the same problem and verify the result.

# Now, let's apply some preprocessing techniques and apply ridge regression on more complex dataset.
x1,x2= dp.random_data()
#print(dp.ridge_randomdata(x1,x2,0.01))





