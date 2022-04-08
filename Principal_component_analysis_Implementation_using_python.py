########### Developed by Nikhil #################

### PCA Implementation using python.

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import  StandardScaler
def sample_data():

    try:
        np.random.seed(50)
        mean = np.array([0,0,0])
        cov = np.array([[1,0,0],[0,1,0],[0,0,1]])
        class1 = np.random.multivariate_normal(mean,cov,50)
        data_class1 = pd.DataFrame(class1,columns=['Sample1','Sample2','Sample3'])
        data_class1['target'] = 1

        ### Now, generate the data for class2 in a same way.

        mean1=np.array([1,1,1])
        cov1=np.array(([1,0,0],[0,1,0],[0,0,1]))
        class2= np.random.multivariate_normal(mean1,cov1,50)
        data_class2=pd.DataFrame(class2,columns=['Sample1','Sample2','Sample3'])
        data_class2['target'] = 0
        df = data_class1.append(data_class2,ignore_index=True)
        return df

    except Exception as e:
        return e
data=sample_data()


def plot_data(data):
    fig =  px.scatter_3d(data,x=data['Sample1'],y=data['Sample2'],z=data['Sample3'],color=data['target'].astype('str'))
    fig.show()
plot_data(data)

############# Step-2 : Scale the dataset ####################

def scaling(data):
    scaler = StandardScaler()
    trf_data =  scaler.fit_transform(data.iloc[:,0:3])
    return trf_data

trf_data = scaling(data)


####### Step3 - Find covariance matrix. ##############

def cov_mat(data):
    covariance_matrix = np.cov([data.iloc[:,0],data.iloc[:,1],data.iloc[:,2]])
    return covariance_matrix

covariance_matrix=cov_mat(data)

##### Step4 - Perform the Covariance matrix decomposition where we need to find the eigenvalues and eigenvectors.

def eigen_val(covariance_matrix):

    '''Eigenvectors : These are are special vectors, after applying linear transformation
    these vectors do not change its direction.
    we get one vector in 1-D, 2 in 2D and so on.
    Eigenvalues : The magnitude of Eigenvectors is called as eigenvalues. Also, it finds the new axis where out variance is maximum.'''

    eigenvalues = np.linalg.eig(covariance_matrix)[0]
    return eigenvalues

eigenvalues =eigen_val(covariance_matrix)


def eigen_vec(covariance_matrix):
    eigenvectors = np.linalg.eig(covariance_matrix)[1]
    return eigenvectors
eigenvectors = eigen_vec(covariance_matrix)


### Step5 - Find the principal componets for that our variance is maximum. Now, we will project the datapoints on or new axis(vectors)

def PC_(eigenvectors,data):
    pc = eigenvectors[0:2]
    transformed_df =  np.dot(data.iloc[:,0:3],pc.T)
    df = pd.DataFrame(transformed_df,columns=['PC1','PC2'])
    df['target'] = data['target'].values
    return df

extracted_data = PC_(eigenvectors,data)
print(extracted_data)

def plot_extracted_data(extracted_data):
    fig = px.scatter(extracted_data, x=extracted_data['PC1'], y=extracted_data['PC2'],color=extracted_data['target'].astype('str'))
    fig.show()
plot_extracted_data(extracted_data)