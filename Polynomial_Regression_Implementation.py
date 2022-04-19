##### Developed by : Nikhil #################
'''

Polynomial Regression : It is a special form of linear regression that we use in the case where our data is not linear.

- In this algorithm we apply the same steps we perform in linear regression. This only difference is that, in this technique
we increase the degree of our features and regression algorithm applied on the features.

- After applying this technique, our model try to capture the non-linear properties in the dataset.

- However, it is very crucial to choose right degree in this regression technique. Otherwise, our model will be overfit
in case of higher degree and underfit if degree is low.

- Let's support we have two features: X|y   where x = 5 ,y =2 . if increase the degree of x to 2 , our feature values would
be : x**0,x**1,x**2 : 1,2,4 respective.(** shows x power 0 and vice-versa)

- Note - In this technique, we only increase the degree of input feature and output feature remains same.

- why do we call it Linear regression ?

# Because the degree of coefs. is still same and relationship between y and coefs. is till linear. That's why we call it
form of Linear regression.

# Bias-variance tradeoff

-Bias : The inability of a ML model to truly capture the relationship in the training data. High bias.
-Variance : If a model is giving high  % of errors for test set and for training is low. High variance.

- high bias and high variance : Underfitting
- low bias,high variance : Overfitting
- Low bias , low variance : Generalised model

'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.metrics import  r2_score
from sklearn.linear_model import  LinearRegression

### First generate the non-linear data
class data:

    def Nonlinear_data(self):
        X = 6 * np.random.rand(200, 1) - 3
        y = 0.8 * X**2 + 0.9 * X + 2 + np.random.randn(200, 1)
        ## Verify by plotting the data.
        '''
        #plt.plot(X,y,'b*')
        #plt.xlabel("X-data")
        #plt.ylabel("Y_data")
        #plt.show()
        '''
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
        return X_train,X_test,y_train,y_test


    def poly_features(self,X_train,X_test):
        pf = PolynomialFeatures(degree=2,include_bias=False) # include_bias=False (it will remove the x0 from the transformed data)
        X_train_trf = pf.fit_transform(X_train)
        X_test_trf = pf.transform(X_test)
        return X_train_trf,X_test_trf,


class regression:

    def lr_(self,X_train,X_test,y_train,y_test):
        lr=LinearRegression()
        lr.fit(X_train,y_train)
        y_pred=lr.predict(X_test)
        r2 = r2_score(y_test,y_pred)
        return "R2-Score Before poly features:",  r2

    def lr_2(self, X_train_trf, X_test_trf, y_train, y_test):
        lr = LinearRegression()
        lr.fit(X_train_trf, y_train)
        y_pred = lr.predict(X_test_trf)
        r2 = r2_score(y_test, y_pred)
        return "R2-Score After poly features:", r2


data = data()
X_train,X_test,y_train,y_test = data.Nonlinear_data()
X_train_trf,X_test_trf = data.poly_features(X_train,X_test)
reg = regression()
print(reg.lr_(X_train,X_test,y_train,y_test))
print(reg.lr_2(X_train_trf,X_test_trf,y_train,y_test))

## We can see after transforming the features R2 Score increased from 10-30% to 80-95%.


