"""
PROJECT_NAME = Datawhale
Author : sciengineer
Email : 821072960@qq.com
Time = 2021/7/15 13:36
"""
import numpy as np
from sklearn import datasets

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use a part of features (for example:feature 2~5)
diabetes_X = diabetes_X[:, 2:6]

# The training features
diabetes_X_train = diabetes_X[:-20]
# The training label
diabetes_y_train = diabetes_y[:-20]

# linear regression model : y_hat = theta0 + theta1 * x1 + theta2 * x2 + ... + thetan * xn

# features:x1, x2, ... , xn
x = diabetes_X_train
# label : y
y = diabetes_y_train

# learning rate: lr
lr = 0.001


# gradient descent algorithm
def grad_desc(x_ndarr, y_ndarr):


    """
    :param x_ndarr: ndarray of features (for example, 422 samples with 3 fetures, then the shape of x_ndarr
    is (422,3)
    :param y_ndarr: ndarray of label (for example, 422 samples with 1 label, then the shape of y_ndarr
    is (422,)
    :return: a list contains interception and coeffients of the linear regression model
    """
    x0 = np.ones(x_ndarr.shape[0])
    # in order to use matrix multiplication, add this column x0
    x_ndarr = np.insert(x_ndarr, 0, x0, axis=1)
    theta_arr = np.zeros(x_ndarr.shape[1])
    # to find a reasonable number to initialize the j_loss_last, I
    # calculate the j_loss when i == 0 in the for loop (12435769.0), and
    # set j_loss_last the  same order of magnitude with j_loss.
    j_loss_last = 1e7

    for i in range(10 ** 10):
        # y_hat = theta0 * x0 + theta1 * x1 +theta2 * x2+ ... +thetan * xn
        y_hat = np.dot(x_ndarr, theta_arr)
        j_loss = np.dot((y_hat - y_ndarr).T, (y_hat - y_ndarr))
        delta_j_loss = j_loss_last - j_loss
        rate = abs(delta_j_loss / j_loss)
        # partial derivative of function j_loss with respect to variable theta_arr
        pd_j2theta_arr = np.dot(y_hat - y_ndarr, x_ndarr)
        # theta_arr updates each interation
        theta_arr = theta_arr - lr * pd_j2theta_arr
        j_loss_last = j_loss
        print('epoch:{}, loss:{}'.format(i, j_loss))
        # I choose the rate as the condition of convergence
        if rate < 1e-12:
            break
    return theta_arr


if __name__ == '__main__':
    theta_arr = grad_desc(x, y)
    # The coeffients: theta1, theta2,..., thetan
    print('Coeffients: \n', theta_arr[1:])
    # The interception: theta0
    print('Interception: \n', theta_arr[0])
