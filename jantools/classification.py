"""
===============================================================================
 Script Name   : classification.py
 Author        : Jan
 Created       : 2025-09-25
 Version       : 1.0

 Description   :
    Self-implemented methods for classification problems
===============================================================================
"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def f_wb(w,x,b):
    return sigmoid(np.dot(w,x) + b)

def loss(x,y,w,b):
    return -y * np.log(f_wb(w, x, b)) - (1 - y) * np.log(1 - f_wb(w, x, b))

def compute_cost(w,b,x,y):
    m, n = x.shape

    sum=0
    for i in range(m):
        sum = sum + loss(x[i],y[i],w,b)

    return sum/m

def compute_gradient(x,y,w,b):
    dj_db = 0
    dj_dw = 0
    m, n = x.shape

    for i in range(m):
        dj_db = dj_db + (f_wb(w,x[i],b) - y[i])

        for j in range(n):
            dj_dw = dj_dw + (f_wb(w,x[i],b) - y[i])*x[i,j]

    return (dj_db/m),(dj_dw/m)

def sigmoid(z):
    if isinstance(z, np.ndarray):
        g = [0] * len(z)
        for i in range(0, len(z)):
            g[i] = 1 / (1 + np.exp(-1 * z[i]))
    else:
        g = 0;
        g = 1 / (1 + np.exp(-1 * z))

    return g

def gradient_descent(x,y,w_in,b_in,alpha,num_iters):
    m = len(x)

    J = []
    w = []

    for i in range(num_iters):
        dj_db, dj_dw = compute_gradient(x,y,w_in,b_in)

        w_in = w_in - alpha * dj_dw
        b_in = b_in - alpha * dj_db

        if i<100000:
            cost = compute_cost(w_in,b_in,x,y)
            J.append(cost)
            w.append(w_in)

    return w_in, b_in, J, w