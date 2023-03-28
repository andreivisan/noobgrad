import numpy as np


def z(X, W, b):
    return np.dot(X, W) + b

'''
SIGMOID
'''
# Define the sigmoid activation function
def sigmoid(X, W, b):
    f_x = z(X, W, b)
    return 1 / (1 + np.exp(-f_x))

def logistic_regression_derivatives(X, y, w, b):
    samples_size = X.shape[0]
    
    # Compute predicted probabilities
    p = sigmoid(X,w,b)
    
    # Compute the derivatives(gradients) for w and b
    dW = (1 / samples_size) * np.dot(X.T, (p - y))
    db = (1 / samples_size) * np.sum(p - y)
    
    return dW, db

def train_logistic_regression(X, y, learning_rate, epochs):
    samples_size, features_size = X.shape
    
    # Initialize the weights and bias
    W = np.zeros((features_size, 1))
    b = 0
    
    for epoch in range(epochs):
        # Compute the derivatives(gradients) for w and b
        dW, db = logistic_regression_derivatives(X, y, W, b)
        
        # Update the weights and bias
        W -= learning_rate * dW
        b -= learning_rate * db
        
        # Optional: Compute and print loss every 10 epochs
        if epoch % 10 == 0:
            p = sigmoid(X,W,b)
            loss = -1 / samples_size * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
            print(f'Epoch {epoch}/{epochs}, Loss: {loss}')
        
    return W, b