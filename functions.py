import numpy as np
def sigmoid(X):
    return 1.0 / (1 + np.exp(-X))

def computeCost(X, y, Theta):
    m = np.size(y)
    z = sigmoid(X @ Theta)
    J = ((np.log(z) @ (-y)) - (np.log(1 - z) @ (1 - y))) / m
    return J

def computeGradient(X, y, Theta, m):
    return (X.T @ (sigmoid(X @ Theta) - y)) / m
def GradientDescent(X, y, Theta, alpha=0.1, iter=10000):
    m = X.shape[0]
    J_hist = np.zeros((iter, 2))
    for i in range(iter):
        Theta -= alpha * computeGradient(X,y,Theta,m)
        cost = computeCost(X, y, Theta)
        J_hist[i, 0] = i
        J_hist[i, 1] = cost
    return Theta, J_hist

def normalize(X):
    return (X - np.mean(X,axis=0)) / (np.max(X) - np.min(X)).values

def predict(ypred, threshold=0.5):
    m = len(ypred)
    for i in range(m):
        if ypred[i] < threshold:
            ypred[i] = 0
        else:
            ypred[i] = 1
    return ypred

def acc(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    return 1.0 * correct / y_true.shape[0]