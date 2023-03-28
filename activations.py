import numpy as np


def f_x(X, W, b):
    return np.dot(X, W) + b


def compute_gradients(X, y, w, b, model):
    samples_size = X.shape[0]
    
    # Compute predicted values
    p = model
    
    # Compute the derivatives(gradients) for w and b
    dW = (1 / samples_size) * np.dot(X.T, (p - y))
    db = (1 / samples_size) * np.sum(p - y)
    
    return dW, db


def train_model(X, y, W, b, learning_rate, epochs, model) :
    samples_size = X.shape[0]
    
    for epoch in range(epochs):
        # Compute the derivatives(gradients) for w and b
        dW, db = compute_gradients(X, y, W, b, model)
        
        # Update the weights and bias
        W -= learning_rate * dW
        b -= learning_rate * db
        
        # Optional: Compute and print loss every 10 epochs
        if epoch % 10 == 0:
            p = sigmoid(X,W,b)
            loss = -1 / samples_size * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
            print(f'Epoch {epoch}/{epochs}, Loss: {loss}')
        
    return W, b


'''
LINEAR REGRESSION
'''
def linear(X, W, b):
    return f_x(X, W, b)

def linear_regression_derivatives(X, y, w, b):
    return compute_gradients(X, y, w, b, f_x(X, w, b))

def train_linear_regression(X, y, learning_rate, epochs):
    features_size = X.shape[1]
    # Initialize the weights and bias
    W = np.zeros((features_size, 1))
    b = 0
    return train_model(X, y, W, b, learning_rate, epochs, f_x(X, W, b))


'''
SIGMOID
'''
# Define the sigmoid activation function
def sigmoid(X, W, b):
    z = f_x(X, W, b)
    return 1 / (1 + np.exp(-z))

# Derivatives are also called gradients
def logistic_regression_derivatives(X, y, w, b):
    return compute_gradients(X, y, w, b, sigmoid(X, w, b))

def train_logistic_regression(X, y, learning_rate, epochs):
    features_size = X.shape[1]
    # Initialize the weights and bias
    W = np.zeros((features_size, 1))
    b = 0
    return train_model(X, y, W, b, learning_rate, epochs, sigmoid(X, W, b))