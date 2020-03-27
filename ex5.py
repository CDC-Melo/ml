# divide data into train, cross validation, and test sets.6:2:2

# overfit

# model selection - choose polynomail degree to fit your data

# find the model with the lowest test set error

# the ability of generalization

# Jcv and Jtest

# choose lamada

# learning curves

# bias and variance

# 10-7 evaluation

# error analysis

import numpy as np
import scipy.io as sio
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def cost(theta, X, y):
    """
    :param theta: R(n), linear regression parameters
    :param X: R(m*n), m records, n features
    :param y: R(m)
    :return: cost
    """
    m = X.shape[0]

    inner = np.dot(X, theta) - y
    square_sum = np.dot(inner.T, inner)
    cost = square_sum / (2 * m)

    return cost


def costReg(theta, X, y, reg = 1):
    m = X.shape[0]
    regularized_term = (reg / (2*m)) * np.power(theta[1:], 2).sum()

    return cost(theta, X, y) + regularized_term


def gradient(theta, X, y):
    m = X.shape[0]
    inner = np.dot(X.T, (np.dot(X, theta) - y))
    # X.T->(n, m)  X @ theta -> (m, n) @ (n, 1) -> (m, 1)
    # (n, m) @ (m, 1) -> (n, 1)
    return inner / m


def gradientReg(theta, X, y, reg):
    m = X.shape[0]

    regularized_term = theta.copy()
    regularized_term[0] = 0  # subscript = 0, don't regularize

    regularized_term = (reg / m) * regularized_term

    return  gradient(theta, X, y) + regularized_term


def linear_regression(X, y, l=1):
    """
    :param X: feature matrix, (m, n+1)
    :param y: target vector, (m, )
    :param l: lambda constant for regularization
    :return:trained parameters
    """
    theta = np.ones(X.shape[1])

    res = opt.minimize(fun = costReg,
                       x0 = theta,
                       args=(X, y, l),
                       method='TNC',
                       jac = gradientReg,
                       options={'disp':True})

    return res


def poly_features(x, power, as_ndarray=False):
    data = {'f{}'.format(i): np.power(x, i) for i in range(1, power + 1)}
    df = pd.DataFrame(data)

    return df.values if as_ndarray else df


def normalize_feature(df):
    # Applies function along input axis(default 0) of DataFrame.
    return df.apply(lambda column: (column - column.mean()) / column.std())


def prepare_poly_data(*args, power):
    """
    args: keep feeding in X, Xval, or Xtest
        will return in the same order
    """
    def prepare(x):
        # expand feature
        df = poly_features(x, power=power)

        # normalization
        ndarr = normalize_feature(df).values

        # add intercept term
        return np.insert(ndarr, 0, np.ones(ndarr.shape[0]), axis=1)

    return [prepare(x) for x in args]


def plot_learning_curve(X, Xinit, y, Xval, yval, l=0):
    training_cost, cv_cost = [], []
    m = X.shape[0]

    for i in range(1, m + 1):
        # regularization applies here for fitting parameters
        res = linear_regression(X[:i, :], y[:i], l=l)

        # remember, when you compute the cost here, you are computing
        # non-regularized cost. Regularization is used to fit parameters only
        tc = cost(res.x, X[:i, :], y[:i])
        cv = cost(res.x, Xval, yval)

        training_cost.append(tc)
        cv_cost.append(cv)

    fig, ax = plt.subplots(2, 1, figsize=(12, 12))
    ax[0].plot(np.arange(1, m + 1), training_cost, label='training cost')
    ax[0].plot(np.arange(1, m + 1), cv_cost, label='cv cost')
    ax[0].legend()

    fitx = np.linspace(-50, 50, 100)
    fitxtmp = prepare_poly_data(fitx, power=8)
    fity = np.dot(prepare_poly_data(fitx, power=8)[0], linear_regression(X, y, l).x.T)

    ax[1].plot(fitx, fity, c='r', label='fitcurve')
    ax[1].scatter(Xinit, y, c='b', label='initial_Xy')

    ax[1].set_xlabel('water_level')
    ax[1].set_ylabel('flow')


data = sio.loadmat('ex5data1.mat')
X, y, Xval, yval, Xtest, ytest = map(np.ravel, [data['X'], data['y'], data['Xval'], data['yval'], data['Xtest'], data['ytest']])

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X, y)
ax.set_xlabel('water_level')
ax.set_ylabel('flow')

X, Xval, Xtest = [np.insert(x.reshape(x.shape[0], 1), 0, np.ones(x.shape[0]), axis=1) for x in (X, Xval, Xtest)]

theta = np.ones(X.shape[1])
print(costReg(theta, X, y, 1))

print(gradientReg(theta, X, y, 1))

theta = np.ones(X.shape[1])
final_theta = opt.minimize(fun=costReg, x0=theta, args=(X, y, 0), method='TNC', jac=gradientReg, options={'disp': True}).x

k = final_theta[0]
b = final_theta[1]

fig, ax = plt.subplots(figsize=(12,8))
plt.scatter(X[:,1], y, c='r', label="Training data")
plt.plot(X[:, 1], X[:, 1]*b + k, c='b', label="Prediction")
ax.set_xlabel('water_level')
ax.set_ylabel('flow')
ax.legend()
# plt.show()

training_cost, cv_cost = [], []

m = X.shape[0]
for i in range(1, m + 1):
    res = linear_regression(X[:i, :], y[:i], 0)

    tc = costReg(res.x, X[:i, :], y[:i], 0)
    cv = costReg(res.x, Xval, yval, 0)

    training_cost.append(tc)
    cv_cost.append(cv)

fig, ax = plt.subplots(figsize=(12,8))
plt.plot(np.arange(1, m+1), training_cost, label='training cost')
plt.plot(np.arange(1, m+1), cv_cost, label='cv cost')
plt.legend()

data = sio.loadmat('ex5data1.mat')
X, y, Xval, yval, Xtest, ytest = map(np.ravel, [data['X'], data['y'], data['Xval'], data['yval'], data['Xtest'], data['ytest']])
poly_features(X, power=3)

X_poly, Xval_poly, Xtest_poly= prepare_poly_data(X, Xval, Xtest, power=8)

plot_learning_curve(X_poly, X, y, Xval_poly, yval, l=0)
plot_learning_curve(X_poly, X, y, Xval_poly, yval, l=1)
# plt.show()

l_candidate = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
training_cost, cv_cost = [], []

for l in l_candidate:
    res = linear_regression(X_poly, y, l)

    tc = cost(res.x, X_poly, y)
    cv = cost(res.x, Xval_poly, yval)

    training_cost.append(tc)
    cv_cost.append(cv)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(l_candidate, training_cost, label='training')
ax.plot(l_candidate, cv_cost, label='cross validation')
plt.legend()
plt.xlabel('lambda')
plt.ylabel('cost')
# plt.show()

for l in l_candidate:
    theta = linear_regression(X_poly, y, l).x
    print('test cost(l={}) = {}'.format(l, cost(theta, Xtest_poly, ytest)))