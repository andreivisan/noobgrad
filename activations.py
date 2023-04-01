import numpy as np


def f_x(X, W, b):
    return np.dot(X, W) + b


def compute_gradients(X, y, w, b, activation):
    samples_size = X.shape[0]
    y = y.reshape(-1,1)             # ensure 2D
    w = w.reshape(-1,1)
    
    # Compute predicted values
    if activation == 'sigmoid':
        p = sigmoid(X, w, b)
    elif activation == 'linear':
        p = linear(X, w, b)
    elif activation == 'relu':
        p = relu(X, w, b)
    
    # Compute the derivatives(gradients) for w and b
    err = p - y
    dW = (1 / samples_size) * X.T @ err
    db = (1 / samples_size) * np.sum(err)
    
    return dW, db


# Because we are using epochs, we are not using the cost function
# TODO A future optimization would be to use the cost function to stop the training
# We could keep the epoch count and stop when the cost function is not changing
def train_model(X, y, learning_rate, epochs, activation) :
    samples_size, features_size = X.shape
    
    # Initialize weights and bias
    W = np.zeros((features_size, 1))
    w_in_shape = W.shape
    b = np.zeros((1, 10))
    
    for epoch in range(epochs):
        # Compute the derivatives(gradients) for w and b
        dW, db = compute_gradients(X, y, W, b.T, activation)
        
        # Update the weights and bias
        W -= learning_rate * dW
        b -= learning_rate * db
        
        # Optional: Compute and print loss every 10 epochs
        if epoch % 100 == 0:
            print(f'Epoch {epoch}/{epochs}, dW, db: {dW}, {db}')
        
    return W.reshape(w_in_shape), b

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
As for ReLU we should use backpropagation and a suitable loss function
we will use the MSE loss function and custom gradients and weights and bias update
'''
def relu(X, W, b):
    return np.maximum(0, f_x(X, W, b))

def relu_derivative(x):
    return (x > 0).astype(float)

def mse_loss(y_true, y_pred):
    n_samples = y_true.shape[0]
    return 1 / (2 * n_samples) * np.sum((y_pred - y_true) ** 2)

def train_relu(X, y, learning_rate, epochs):
    return train_model(X, y, learning_rate, epochs, activation='relu')