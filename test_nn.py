import numpy as np
from utils import load_data
from nn import DenseLayer, NeuralNetwork

if __name__ == "__main__":
    # Load data
    X, y = load_data()
    
    print ('The first element of X is: ', X[0])

    n_batches, n_features = X.shape

    print(f'batches: {n_batches} | features: {n_features}')
