import numpy as np
import matplotlib.pyplot as plt

### 基于numpy创建梯度下降 ###

def loss_function(y_hat, y):
    return sum((y_hat - y.reshape(y_hat.shape))**2)/2/len(y)

def para_update(w, b, X, y, lr=0.001):
    y_hat = np.dot(X, w) + b
    w -= np.dot(y_hat - y,X) *lr/len(y)
    b -= np.mean(y_hat - y)
    return w,b


def generate_data(n_examples, w_true, b_true):
    features = np.random.normal(0, 1, size=(n_examples, len(w_true)))
    labels = np.dot(features, w_true) + b_true
    labels += np.random.normal(0,0.02, size=(len(labels),))
    return features, labels

w_true = np.array([0.2,0.4,0.6])
b_true = np.array(10)
X, y = generate_data(100, w_true, b_true)
epoch = 100

w = np.zeros(w_true.shape)
b = 0
lossLs = []
for i in range(epoch):
    y_hat = np.dot(X, w) + b
    w, b = para_update(w, b, X, y)
    loss = loss_function(y_hat, y)
    lossLs.append(loss)
    if i%5 == 0:
        print("epoch:{}, loss:{}".format(i,loss))
print(f"w:{w},b:{b}")



### torch 手写梯度下降 ###

import torch 



w_true = np.array([0.2,0.4,0.6])
b_true = np.array(10)
X, y = generate_data(100, w_true, b_true)
X, y = torch.FloatTensor(X), torch.FloatTensor(y)
epoch = 100


def sgd(params, lr, batch_size):  
    """小批量随机梯度下降。"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
    return param

w = torch.normal(0, 0.01, size=(3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
epoch = 100
lossLs = []
for i in range(epoch):
    y_hat = torch.matmul(X, w) + b
    loss = loss_function(y_hat, y)
    loss.backward()
    para = sgd((w,b), lr=0.1, batch_size=len(y)) 
    lossLs.append(loss)
    if i%5 == 0:
        print("epoch:{}, loss:{}".format(i,loss.item()))
print(f"w:{w},b:{b}")
plt.plot(lossLs)
plt.show()