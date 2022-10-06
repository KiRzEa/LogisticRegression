import numpy as np
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))
def compute_cost(X, y, w, b):
    m, n = X.shape
    total_cost = 0
    for i in range(m):
        error = -y[i] * np.log(sigmoid(X[i] @ w + b)) - (1 - y[i]) * np.log(1 - sigmoid(X[i] @ w + b))
        total_cost += error
    total_cost = total_cost / m
    return total_cost
def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    for i in range(m):
        z_i = sigmoid(X[i] @ w + b)
        error = z_i - y[i]
        for j in range(n):
            dj_dw[j] += error * X[i, j]
        dj_db += error
    dj_dw =dj_dw / m
    dj_db = dj_db / m
    return dj_dw, dj_db
def gradient_descent(X, y, w, b, alpha=1e-4, iter=10000):
    m = X.shape[0]
    J_hist = np.zeros((iter, 2))

    for i in range(iter):
        dj_dw, dj_db = compute_gradient(X, y, w, b)

        w -= alpha * dj_dw
        b -= alpha * dj_db

        cost = compute_cost(X, y, w, b)
        J_hist[i, 0] = i
        J_hist[i, 1] = cost

        # if i % 1000 == 0:
        #     w_hist.append(w)
        #     print(f"Iteration {i:4}: Cost {float(J_hist[-1]):8.2f}   ")

    return w, b, J_hist

def predict(ypred, threshold=0.58):
    m = len(ypred)
    for i in range(m):
        if ypred[i] < threshold:
            ypred[i] = 0
        else:
            ypred[i] = 1
    return ypred