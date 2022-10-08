import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#cp, restecg, slope, ca, thal: multi-class
def getDummies(raw):
    cp = pd.get_dummies(raw['cp'],prefix='cp')
    restecg = pd.get_dummies(raw['restecg'],prefix='restecg')
    ca = pd.get_dummies(raw['ca'],prefix='ca')
    slope = pd.get_dummies(raw['slope'],prefix='slope')
    thal = pd.get_dummies(raw['thal'],prefix='thal')
    return pd.concat((raw,cp,restecg,ca,slope,thal),axis=1)

def sigmoid(X):
    return 1.0 / (1 + np.exp(-X))

def computeCost(X, y, Theta, lambda_):
    m = np.size(y)
    z = sigmoid(X @ Theta)
    J = ((np.log(z) @ (-y)) -  (np.log(1 - z) @ (1 - y))) / m
    return J

def computeGradient(X, y, Theta, m):
    return (X.T @ (sigmoid(X @ Theta) - y)) / m
def GradientDescent(X, y, Theta, alpha=1, iter=1000, lambda_=1):
    m = X.shape[0]
    J_hist = np.zeros((iter, 2))
    for i in range(iter):
        Theta = Theta - alpha * computeGradient(X,y,Theta,m)
        cost = computeCost(X, y, Theta, lambda_)
        J_hist[i, 0] = i
        J_hist[i, 1] = cost
    return Theta, J_hist

def normalize(X):
    return (X - np.mean(X,axis=0)) / np.std(X,axis=0)

def predict(ypred, threshold=0.5):
    m = ypred.shape[0]
    for i in range(m):
        if ypred[i] < threshold:
            ypred[i] = 0
        else:
            ypred[i] = 1
    return ypred

def acc(y, y_pred):
    y_true = np.array(y)
    correct = np.sum(y_true == y_pred)
    return 1.0 * correct / y_true.shape[0]
def f1_score(y, y_pred):
    y_true = np.array(y)
    m = y_true.shape[0]
    TP = 0.
    FP = 0.
    FN = 0.
    for i in range(m):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        if y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        if y_true[i] == 1 and y_pred[i] == 0:
            FN += 1
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1
