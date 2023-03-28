import numpy as np


def f_x(X, W, b):
    return np.dot(X, W) + b


def compute_gradients(X, y, w, b, activation):
    samples_size = X.shape[0]
    
    # Compute predicted values
    if activation == 'sigmoid':
        p = sigmoid(X, w, b)
    else:
        p = linear(X, w, b)
    
    # Compute the derivatives(gradients) for w and b
    dW = (1 / samples_size) * np.dot(X.T, (p - y))
    db = (1 / samples_size) * np.sum(p - y)
    
    return dW, db


def train_model(X, y, learning_rate, epochs, activation) :
    samples_size, features_size = X.shape
    
    # Initialize weights and bias
    W = np.zeros((features_size, 1))
    b = 0
    
    for epoch in range(epochs):
        # Compute the derivatives(gradients) for w and b
        dW, db = compute_gradients(X, y, W, b, activation)
        
        # Update the weights and bias
        W -= learning_rate * dW
        b -= learning_rate * db
        
        # Optional: Compute and print loss every 10 epochs
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, dW, db: {dW}, {db}')
        
    return W, b

'''
LINEAR REGRESSION
'''
def linear(X, W, b):
    return f_x(X, W, b)

def train_linear_regression(X, y, learning_rate, epochs):
    return train_model(X, y, learning_rate, epochs, activation='linear')


'''
SIGMOID
'''
# Define the sigmoid activation function
def sigmoid(X, W, b):
    z = f_x(X, W, b)
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, y, learning_rate, epochs):
    return train_model(X, y, learning_rate, epochs, activation='sigmoid')


'''
ReLU
'''
def relu(X, W, b):
    return np.maximum(0, f_x(X, W, b))