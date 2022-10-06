import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import *
#Manipulating Data
raw = pd.read_csv('heart.csv')
X = raw.drop(['target'],axis=1)
y = raw['target']
X[['age', 'trestbps','chol','thalach','oldpeak']] = normalize(X[['age', 'trestbps','chol','thalach','oldpeak']])
m, n = X.shape
bias_col = np.ones((m, 1))
X = np.hstack((bias_col, X))
#Init parameters
Theta = np.zeros(n + 1)
#Find optimized parameters
Theta, J_hist = GradientDescent(X[:700, :], y[:700], Theta)
#Testing
z = sigmoid(X[700:, :] @ Theta)
predicted = predict(z)
#Accuracy
print(acc(y[700:],z))
#Plot cost function
plt.figure(1)
plt.plot(J_hist[:, 0], J_hist[:, 1])
plt.show()