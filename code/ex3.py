import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import loadmat
from sklearn.metrics import classification_report
from scipy.optimize import minimize


def sigmoid(z):    # sigmoid function
    return 1 / (1 + np.exp(-z))


def cost(theta, X, y, learningRate):  # cost function
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    left = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    right = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    small = (learningRate / (2*len(X))) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    return np.sum(left - right) / len(X) + small


def gradient(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    error = sigmoid(X * theta.T) - y

    grad = ((X.T * error) / len(X)).T +((learningRate / len(X)) * theta)
    grad[0, 0] = np.sum(np.multiply(error, X[:, 0])) / len(X)

    return np.array(grad).ravel()


def one_to_all(X, y, num_labels, learningRate):
    rows = X.shape[0]
    parameters = X.shape[1]

    all_theta = np.zeros((num_labels, parameters + 1))
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    for i in range(1, num_labels + 1):
        theta = np.zeros(parameters + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))

        # minimize the objective function
        fmin = minimize(fun=cost, x0=theta, args=(X, y_i, learningRate), method='TNC', jac=gradient)
        all_theta[i - 1, :] = fmin.x

    return all_theta


def predict_all(X, all_theta):
    rows = X.shape[0]
    params = X.shape[1]
    num_labels = all_theta.shape[0]

    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    X = np.matrix(X)
    all_theta = np.matrix(all_theta)

    h = sigmoid(X * all_theta.T)

    h_argmax = np.argmax(h, axis=1)

    h_argmax = h_argmax + 1

    return h_argmax


data = loadmat('ex3data1.mat')

rows = data['X'].shape[0]
params = data['X'].shape[1]

all_theta = np.zeros((10, params + 1))

X = np.insert(data['X'], 0, values=np.ones(rows), axis=1)

theta = np.zeros(params + 1)

y_0 = np.array([1 if label == 0 else 0 for label in data['y']])
y_0 = np.reshape(y_0, (rows, 1))

all_theta = one_to_all(data['X'], data['y'], 10, 1)
y_pred = predict_all(data['X'], all_theta)
# print(classification_report(data['y'], y_pred))

weight = loadmat("ex3weights.mat")
theta1, theta2 = weight['Theta1'], weight['Theta2']

X2 = np.matrix(np.insert(data['X'], 0, values=np.ones(X.shape[0]), axis=1))
y2 = np.matrix(data['y'])

a1 = X2
z2 = a1 * theta1.T
a2 = sigmoid(z2)

a2 = np.insert(a2, 0, values=np.ones(a2.shape[0]), axis=1)
z3 = a2 * theta2.T

a3 = sigmoid(z3)
y_pred2 = np.argmax(a3, axis=1) + 1
print(classification_report(y2, y_pred))