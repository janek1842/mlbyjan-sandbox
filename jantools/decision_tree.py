"""
===============================================================================
 Script Name   : decision_tree.py
 Author        : Jan
 Created       : 2025-10-24
 Version       : 1.0

 Description   :
    Self-implemented methods for implementing decision trees
===============================================================================
"""
import numpy as np

def compute_entropy(y):
    """

    Args:
       y (ndarray): Numpy array indicating whether each example at a node is
           edible (`1`) or poisonous (`0`)

    """

    entropy = 0.

    counter = 0
    for i in range(len(y)):
        if y[i] == 1:
            counter = counter + 1

    if len(y) == 0 or counter == 0 or counter == len(y):
        entropy = 0
    else:
        p1 = counter / len(y)
        entropy = -1 * p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)

    return entropy


# UNQ_C2
# GRADED FUNCTION: split_dataset

def split_dataset(X, node_indices, feature):

    left_indices = []
    right_indices = []

    for idx, row in enumerate(X):

        if idx in node_indices:

            if row[feature] == 1:
                left_indices.append(idx)

            if row[feature] == 0:
                right_indices.append(idx)

    return left_indices, right_indices

def compute_information_gain(X, y, node_indices, feature):

    # Split dataset
    left_indices, right_indices = split_dataset(X, node_indices, feature)

    # Some useful variables
    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]

    information_gain = 0
    wleft = len(left_indices) / len(node_indices)
    wright = len(right_indices) / len(node_indices)

    information_gain = compute_entropy(y_node) - (wleft * compute_entropy(y_left) + wright * compute_entropy(y_right))

    return information_gain


def get_best_split(X, y, node_indices):

    # Some useful variables
    num_features = X.shape[1]

    # You need to return the following variables correctly
    best_feature = -1
    max_gain = 0

    for feature in range(num_features):

        if compute_information_gain(X, y, node_indices, feature) > max_gain:
            max_gain = compute_information_gain(X, y, node_indices, feature)
            best_feature = feature

    return best_feature


# Not graded
tree = []


def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth):

    # Maximum depth reached - stop splitting
    if current_depth == max_depth:
        formatting = " " * current_depth + "-" * current_depth
        #print(formatting, "%s leaf node with indices" % branch_name, node_indices)
        return

    best_feature = get_best_split(X, y, node_indices)
    formatting = "-" * current_depth
    #print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))

    # Split the dataset at the best feature
    left_indices, right_indices = split_dataset(X, node_indices, best_feature)

    tree.append((left_indices, right_indices, best_feature))

    # continue splitting the left and the right child. Increment current depth
    build_tree_recursive(X, y, left_indices, "Left", max_depth, current_depth + 1)
    build_tree_recursive(X, y, right_indices, "Right", max_depth, current_depth + 1)

    return tree