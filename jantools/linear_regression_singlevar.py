"""
===============================================================================
 Script Name   : linear_regression_singlevar.py
 Author        : Jan
 Created       : 2025-09-23
 Version       : 1.0

 Description   :
    Self-implemented methods for computing linear regression using gradient descent
===============================================================================
"""

def f_wb(w,x,b):
    return w*x + b

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

        print("##########")
        step_w = compute_gradient(x,y,w,b)[0]
        w = w - alpha*step_w
        grad_step_w.append(w)
        print("Iteration: ", i," gradient step w: ",step_w)

        step_b=compute_gradient(x,y,w,b)[1]
        b = b - alpha*step_b
        grad_step_b.append(b)
        print("Iteration: ", i, " gradient step b: ", step_b)

        computed_costs.append(compute_cost(w,y,x,b))
        print("##########")


    return w,b,grad_step_w,grad_step_b,computed_costs