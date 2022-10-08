import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functions import *
#Manipulating Data
raw = pd.read_csv('heart.csv')
raw = getDummies(raw)
raw.drop(columns = ['cp','restecg','slope','ca','thal'], inplace=True)
X = raw.drop(['target'], axis=1)
X[['age', 'trestbps','chol','thalach','oldpeak']] = normalize(X[['age', 'trestbps','chol','thalach','oldpeak']])
y = raw['target']
#526 True Positive
#499 True Negative
id_pos = np.where(y.values.reshape(-1) == 1)[0]
id_neg = np.where(y.values.reshape(-1) == 0)[0]
#Split data rate 7:3
pos_train = X.iloc[id_pos[:368]]
neg_train = X.iloc[id_neg[:349]]
pos_test = X.iloc[id_pos[368:]]
neg_test = X.iloc[id_neg[349:]]
X_train = np.concatenate((pos_train, neg_train),axis=0)
X_test = np.concatenate((pos_test, neg_test),axis=0)
y_train = np.concatenate((y.iloc[id_pos[:368]], y.iloc[id_neg[:349]]),axis=0)
y_test = np.concatenate((y.iloc[id_pos[368:]], y.iloc[id_neg[349:]]),axis=0)
#-------------------------------------------------------
m_train, n_train = X_train.shape
m_test, n_test = X_test.shape
bias_col_train = np.ones((m_train, 1))
bias_col_test = np.ones((m_test, 1))
X_train = np.hstack((bias_col_train, X_train))
X_test = np.hstack((bias_col_test, X_test))
#Init parameters
Theta = np.zeros(n_train + 1)
#Find optimized parameters
Theta, J_hist = GradientDescent(X_train, y_train, Theta)
#Testing
z = sigmoid(X_test @ Theta)
predicted = predict(z)
print(pd.DataFrame(predicted).value_counts())
#F1_score
precision, recall, f1 = f1_score(y_test, z)
print(f"Precision: {precision}\nRecall: {recall}\nF1 score: {f1}")
#Plot cost function
plt.figure(1)
plt.plot(J_hist[:, 0], J_hist[:, 1])
plt.show()