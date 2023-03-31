import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from nn import DenseLayer, NeuralNetwork

if __name__ == "__main__":
   # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Make sure the input data is in the correct shape
    print("X_train shape:", X_train.shape)  # Should be (105, 4)
    print("X_test shape:", X_test.shape)    # Should be (45, 4)
    print("y_train shape:", y_train.shape)  # Should be (105,)
    print("y_test shape:", y_test.shape)    # Should be (45,)


    # Scale the input features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # # Create and train the neural network
    layers = [
        DenseLayer(4, 10, 'relu'),
        DenseLayer(10, 3, 'sigmoid')
    ]
    nn = NeuralNetwork(layers)
    nn.train(X_train, y_train, learning_rate=0.1, epochs=1000)

    # # Test the neural network
    y_pred = nn.forward(X_test)
    # y_pred = np.argmax(y_pred, axis=1)

    # # Calculate accuracy
    # accuracy = accuracy_score(y_test, y_pred)
    # print("Accuracy:", accuracy)

