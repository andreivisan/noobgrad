import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nn import DenseLayer, NeuralNetwork

if __name__ == "__main__":
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the neural network architecture
    layer1 = DenseLayer(n_inputs=X.shape[1], n_neurons=16, activation='relu')
    layer2 = DenseLayer(n_inputs=layer1.weights.shape[1], n_neurons=8, activation='relu')
    layer3 = DenseLayer(n_inputs=layer2.weights.shape[1], n_neurons=len(np.unique(y)), activation='sigmoid')
    network = NeuralNetwork([layer1, layer2, layer3])

    # Train the neural network
    network.train(X_train, y_train, learning_rate=0.1, epochs=1000)

    # Evaluate the performance of the neural network on the testing set
    y_pred = np.argmax(network.forward(X_test), axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: ", accuracy)

