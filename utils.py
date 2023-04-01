import numpy as np


def load_data():
    X = np.load("data/X.npy")
    y = np.load("data/y.npy")
    return X, y