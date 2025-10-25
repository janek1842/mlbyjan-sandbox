"""
===============================================================================
 Script Name   : unsupervized_intro.py
 Author        : Jan
 Created       : 2025-10-24
 Version       : 1.0

 Description   :
    Self-implemented methods for implementing unsupervized ML methods
===============================================================================
"""
import numpy as np
from scipy.stats import multivariate_normal

# UNQ_C1
# GRADED FUNCTION: find_closest_centroids

def find_closest_centroids(X, centroids):
    idx = np.zeros(X.shape[0], dtype=int)

    for i in range(len(X)):
        temp=0
        for ind, u in enumerate(centroids):

            dist = np.linalg.norm(X[i] - u)
            if ind == 0:
                temp = dist
            elif dist < temp:
                temp = dist
                idx[i] = ind

    return idx

def compute_centroids(X, idx, K):

    m, n = X.shape
    centroids = np.zeros((K, n))

    for k in range(K):

        points = []
        for indx, i in enumerate(idx):
            if i == k:
                points.append(indx)

        for s in points:
            centroids[k] += X[s]

        centroids[k] = centroids[k] / len(points)

    return centroids


def kMeans_init_centroids(X, K):

    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])

    # Take the first K examples as centroids
    centroids = X[randidx[:K]]

    return centroids

def estimate_gaussian(X):
    m, n = X.shape
    mu = np.zeros(n)
    var = np.zeros(n)

    for i in range(n):
        suma = 0
        for j in range(0, m):
            suma += X[j, i]

        mu[i] = suma / m

    for i in range(n):
        suma = 0
        for j in range(0, m):
            suma += (X[j, i] - mu[i]) ** 2

        var[i] = suma / m

    return mu, var

def select_threshold(y_val, p_val):

    best_epsilon = 0
    best_F1 = 0
    F1 = 0

    step_size = (max(p_val) - min(p_val)) / 1000

    pred = np.zeros(len(p_val))

    for epsilon in np.arange(min(p_val), max(p_val), step_size):

        ### START CODE HERE ###
        tp, fp, fn = 0, 0, 0
        for ind, p in enumerate(p_val):

            if p < epsilon:
                pred[ind] = 1
            else:
                pred[ind] = 0

        for i in range(len(p_val)):

            if y_val[i] == 1 and pred[i] == 1:
                tp = tp + 1

            if y_val[i] == 0 and pred[i] == 1:
                fp = fp + 1

            if y_val[i] == 1 and pred[i] == 0:
                fn = fn + 1

        if (tp + fp) != 0:
            prec = tp / (tp + fp)
        else:
            prec = 0

        if (tp + fn) != 0:
            rec = tp / (tp + fn)
        else:
            rec = 0

        if (prec + rec) != 0:
            F1 = (2 * prec * rec) / (prec + rec)
        else:
            F1 = 0

        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon

    return best_epsilon, best_F1

def multivariate_gaussian(X, mu, Sigma):

    return multivariate_normal.pdf(X, mean=mu, cov=Sigma)