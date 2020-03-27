import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt


def sigmoid(z):  # sigmoid function
    return 1 / (1 + np.exp(-z))


def cost(theta, X, y):  # cost function
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    left = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    right = np.multiply((1-y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(left - right) / len(X)


def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        grad[i] = np.sum(term) / len(X)

    return grad


def hfunc1(theta, X):
    return sigmoid(np.dot(theta.T, X))


def hfunc2(theta, x1, x2):
    temp = theta[0][0]
    place = 0
    for i in range(1, degree+1):
        for j in range(0, i+1):
            temp+= np.power(x1, i-j) * np.power(x2, j) * theta[0][place+1]
            place+=1
    return temp


def find_decision_boundary(theta):
    t1 = np.linspace(-1, 1.5, 1000)
    t2 = np.linspace(-1, 1.5, 1000)

    cordinates = [(x, y) for x in t1 for y in t2]
    x_cord, y_cord = zip(*cordinates)
    h_val = pd.DataFrame({'x1':x_cord, 'x2':y_cord})
    h_val['hval'] = hfunc2(theta, h_val['x1'], h_val['x2'])

    decision = h_val[np.abs(h_val['hval']) < 2 * 10**-3]
    return decision.x1, decision.x2


def predict(theta, X):
    probability = sigmoid( X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


def costReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    return np.sum(first - second) / len(X) + reg


def gradientReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:, i])

        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[:, i])

    return grad

path = 'ex2data1.txt'  # filepath
data = pd.read_csv(path, header=None, names=['Exam1', 'Exam2', 'Admitted'])  # readfile

data.insert(0, 'Ones', 1)

cols = data.shape[1]  # columns
X = data.iloc[:, 0:cols-1]
y = data.iloc[:, cols-1:cols]
theta = np.zeros(3)
X = np.array(X.values)
y = np.array(y.values)

result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
cost(result[0], X, y)

positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]

plotting_x1 = np.linspace(30, 100, 100)
plotting_h1 = (- result[0][0] - result[0][1] * plotting_x1) / result[0][2]

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(plotting_x1, plotting_h1, 'y', label='Prediction')
ax.scatter(positive['Exam1'], positive['Exam2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam1'], negative['Exam2'], s=50, c='r', marker='x', label='Rejected')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
# plt.show()

theta_min = np.matrix(result[0])
predictions = predict(theta_min, X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
# print('accuracy = {0}%'.format(accuracy))

path2 = 'ex2data2.txt'  # filepath
data2 = pd.read_csv(path2, header=None, names=['Test1', 'Test2', 'Accepted'])  # readf

positive2 = data2[data2['Accepted'].isin([1])]
negative2 = data2[data2['Accepted'].isin([0])]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive2['Test1'], positive2['Test2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative2['Test1'], negative2['Test2'], s=50, c='r', marker='x', label='Rejected')
ax.legend()
ax.set_xlabel('Test1 Score')
ax.set_ylabel('Test2 Score')
# plt.show()

degree = 6
x1 = data2['Test1']
x2 = data2['Test2']

data2.insert(3, 'Ones', 1)

for i in range(1, degree+1):
    for j in range(0, i+1):
        data2['F' + str(i-j) + str(j)] = np.power(x1, i-j) * np.power(x2, j)

data2.drop('Test1', axis=1, inplace=True)
data2.drop('Test2', axis=1, inplace=True)

cols = data2.shape[1]
X2 = data2.iloc[:, 1:cols]
y2 = data2.iloc[:, 0:1]
theta2 = np.zeros(cols-1)

X2 = np.array(X2.values)
y2 = np.array(y2.values)

learningRate = 1

costReg(theta2, X2, y2, learningRate)

result2 = opt.fmin_tnc(func=costReg, x0=theta2, fprime=gradientReg, args=(X2, y2, learningRate))

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive2['Test1'], positive2['Test2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative2['Test1'], negative2['Test2'], s=50, c='r', marker='x', label='Rejected')
ax.set_xlabel('Test1 Score')
ax.set_ylabel('Test2 Score')

x, y = find_decision_boundary(result2)
plt.scatter(x, y, c='y', s=10, label='Prediction')
ax.legend()
plt.show()