"""
===============================================================================
 Script Name   : linear_regression_extended.py
 Author        : Jan
 Created       : 2025-09-25
 Version       : 1.0

 Description   :
    Self-implemented methods for computing linear regression using gradient descent
===============================================================================
"""
import numpy as np

def f_wb(w,x,b):
    return np.dot(w,x) + b

