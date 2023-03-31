import numpy as np
from nn import DenseLayer, NeuralNetwork


if __name__ == "__main__":
    # Prepare the dataset
    X_train = np.random.randn(100, 10)
    y_train = np.random.randn(100, 1)
    X_test = np.random.randn(50, 10)
    y_test = np.random.randn(50, 1)

    # Create the neural network
    layer1 = DenseLayer(10, 5, 'relu')
    layer2 = DenseLayer(5, 1, 'linear')
    network = NeuralNetwork([layer1, layer2])

    # Train the neural network
    learning_rate = 0.01
    epochs = 1000
    network.train(X_train, y_train, learning_rate, epochs)

    # Test the neural network
    y_pred = network.forward(X_test)

    # Evaluate the performance of the neural network
    mse = np.mean((y_test - y_pred)**2)
    print(f'Mean squared error: {mse:.4f}')