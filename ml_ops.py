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
            print(f'Epoch {epoch}/{epochs}, dW, db: {dW}, {db}')
        
    return W, b