"""
===============================================================================
 Script Name   : neural_networks_intro.py
 Author        : Jan
 Created       : 2025-10-16
 Version       : 1.0

 Description   :
    Self-implemented methods for implementing neural networks and compare them with keras/tensorflow
===============================================================================
"""
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def my_dense(a_in, W, b, g):

    units = W.shape[1]
    a_out = np.zeros(units)

    for j in range(units):
        w = W[:, j]
        z = np.dot(w, a_in) + b[j]
        a_out[j] = g(z)

    return a_out

def my_sequential(x, W1, b1, W2, b2, W3, b3):
    a1 = my_dense(x,  W1, b1, sigmoid)
    a2 = my_dense(a1, W2, b2, sigmoid)
    a3 = my_dense(a2, W3, b3, sigmoid)
    return(a3)

def vectorized_dense(a_in, W, b, g):
    a_out = g(np.matmul(a_in, W) + b)
    return a_out

def my_sequential_v(X, W1, b1, W2, b2, W3, b3):
    A1 = vectorized_dense(X,  W1, b1, sigmoid)
    A2 = vectorized_dense(A1, W2, b2, sigmoid)
    A3 = vectorized_dense(A2, W3, b3, sigmoid)
    return(A3)
