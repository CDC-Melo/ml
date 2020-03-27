import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def computeCost(X, y, theta): # J(theta)
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


def gradientDescent(X, y, theta, alpha, iters):  # gradientDescent
    temp = np.matrix(np.zeros(theta.shape))  # theta matrix
    parameters = int(theta.ravel().shape[1])  # flat
    cost = np.zeros(iters)  # cost matrix

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost


def normalization(X, y):
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta





path = 'ex1data1.txt'  # filepath

data = pd.read_csv(path, header=None, names=['Population','Profit'])  # readfile

# data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))  # plot
# plt.show()

data.insert(0, 'Ones', 1)

cols = data.shape[1]
X = data.iloc[:, 0: cols-1]  # features
y = data.iloc[:, cols-1:cols]  # result

X = np.matrix(X.values)
y = np.matrix(y.values)  # numpy matrix
theta = np.matrix(np.array([0, 0]))

# print(computeCost(X, y, theta))
alpha = 0.01
iters = 1500

mytheta, cost = gradientDescent(X, y, theta, alpha, iters)

predict1 = [1, 3.5] * mytheta.T  #two predictions
predict2 = [1, 7] * mytheta.T

x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = theta[0, 0] + (theta[0, 1] * x)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, f, 'g', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
# plt.show()




path2 = 'ex1data2.txt'  # read file
data2 = pd.read_csv(path2, header=None, names=['Size', 'Bedrooms', 'Price'])

data2 = (data2 - data2.mean()) / data2.std()  # normalization
data2.insert(0, 'Ones', 1)

cols2 = data2.shape[1]
X2 = data2.iloc[:, 0:cols2-1]
y2 = data2.iloc[:, cols2-1:cols2]

X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0, 0, 0]))

mytheta2, cost2 = gradientDescent(X2, y2, theta2, alpha,iters)

another_mytheta1 = normalization(X, y)
print(another_mytheta1)