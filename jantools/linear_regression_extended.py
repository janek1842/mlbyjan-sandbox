"""
===============================================================================
 Script Name   : linear_regression_extended.py
 Author        : Jan
 Created       : 2025-09-25
 Version       : 1.0

 Description   :
    Self-implemented methods for computing linear regression using gradient descent for multiple variables
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
    return np.dot(w,x) + b

def compute_cost(w,y,x,b):
    m = len(x)

    cost=0
    for i in range(m):
        cost += (f_wb(w,x[i],b)-y[i])**2
    cost = cost/(2*m)

    return cost

def compute_gradient(x,y,w,b):
    m = len(x)

    dj_dw = 0
    dj_db = 0

    for i in range(m):
        dj_dw = dj_dw + (f_wb(w,x[i],b)-y[i])*x[i]
        dj_db = dj_db + (f_wb(w,x[i],b)-y[i])

    return dj_dw/m,dj_db/m

def run_gradient_descent(x,y,w_in,b_in,alpha,num_iterations):

    w = w_in
    b = b_in

    grad_step_w = []
    grad_step_b = []

    computed_costs = []

    for i in range(num_iterations):

        step_w = compute_gradient(x,y,w,b)[0]
        w = w - alpha*step_w
        grad_step_w.append(w)

        step_b=compute_gradient(x,y,w,b)[1]
        b = b - alpha*step_b
        grad_step_b.append(b)

        computed_costs.append(compute_cost(w,y,x,b))

    return w,b,grad_step_w,grad_step_b,computed_costs

def polynomial_scikit(X,y):
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()

    sgdr_costs = []
    y_pred = []

    for i in range(50):
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        sgdr_costs.append(mean_squared_error(y, y_pred) * 0.5)

    return y_pred, sgdr_costs