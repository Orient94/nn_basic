import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim


def cost_function(X,Y,theta):
    predictions = np.dot(X,theta.T)
    cost = (1/len(Y)) * np.sum((predictions - Y) ** 2)
    return cost



def batch_gradient_descent(X,Y,theta,alpha,iters):
    cost_history = [0] * iters  # 初始化历史损失列表
    for i in range(iters):         
        prediction = np.dot(X,theta.T)                  
        theta = theta - (alpha/len(Y)) * np.dot(prediction - Y,X)   
        cost_history[i] = cost_function(X,Y,theta)               
    return theta,cost_history



if __name__ == '__main__':

    data = load_boston()

    df = pd.DataFrame(data['data'],columns=data['feature_names'])
    df.insert(13,'target',data['target'])
    df.head(5)

    #X,y = df.drop('target',axis=1),df['target']
    X = np.arange(300).reshape(100,3)
    #X = np.random.normal(0,1, size=(100, 3))
    y = X[:,0]*3+X[:,1]*4+X[:,2]*2

    thetas = np.zeros(X.shape[1])
    X_norm = (X - X.min()) / (X.max() - X.min())
    X = X_norm
    batch_theta,batch_history = batch_gradient_descent(X,y,thetas,0.05,500)
    print(batch_theta)
    plt.plot(batch_history)
    plt.show()

