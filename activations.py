import numpy as np
import ml_ops


'''
LINEAR REGRESSION
'''
def linear(X, W, b):
    return ml_ops.f_x(X, W, b)

def linear_regression_derivatives(X, y, w, b):
    return ml_ops.compute_gradients(X, y, w, b, ml_ops.f_x(X, w, b))

def train_linear_regression(X, y, learning_rate, epochs):
    features_size = X.shape[1]
    # Initialize the weights and bias
    W = np.zeros((features_size, 1))
    b = 0
    return ml_ops.train_model(X, y, W, b, learning_rate, epochs, ml_ops.f_x(X, W, b))


'''
SIGMOID
'''
# Define the sigmoid activation function
def sigmoid(X, W, b):
    z = ml_ops.f_x(X, W, b)
    return 1 / (1 + np.exp(-z))

# Derivatives are also called gradients
def logistic_regression_derivatives(X, y, w, b):
    return ml_ops.compute_gradients(X, y, w, b, sigmoid(X, w, b))

def train_logistic_regression(X, y, learning_rate, epochs):
    features_size = X.shape[1]
    # Initialize the weights and bias
    W = np.zeros((features_size, 1))
    b = 0
    return ml_ops.train_model(X, y, W, b, learning_rate, epochs, sigmoid(X, W, b))


'''
ReLU
'''
def relu(X, W, b):
    return np.maximum(0, ml_ops.f_x(X, W, b))