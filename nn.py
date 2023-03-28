import numpy as np
import activations

class DenseLayer:
    def __init__(self, n_inputs, n_neurons, activation):
        '''
        This line of code initializes the weights using a technique called "He initialization" 
        (also known as "He normal initialization"), which is named after Kaiming He, 
        who proposed it in the 2015 paper 
        Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
        '''
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        self.bias = np.zeros((1, n_neurons))
        self.activation = activation
        
    def forward(self, inputs):
        if self.activation == 'linear':
            return activations.linear(inputs, self.weights, self.bias)
        elif self.activation == 'sigmoid':
            return activations.sigmoid(inputs, self.weights, self.bias)
        else:
            raise ValueError(f'Unsupported activation: {self.activation}')
        
class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        
    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    
    def train(self, X, y, learning_rate, epochs):
        if len(self.layers) != 1:
            raise ValueError("Training is only supported for single-layer networks.")
        
        if self.layers[0].activation == 'linear':
            self.layers[0].weights, self.layers[0].bias = activations.train_linear_regression(X, y, learning_rate, epochs)
        elif self.layers[0].activation == 'sigmoid':
            self.layers[0].weights, self.layers[0].bias = activations.train_logistic_regression(X, y, learning_rate, epochs)
            

if __name__ == "__main__":
    # Example usage
    X = np.random.randn(100, 2)
    y = (np.random.rand(100, 1) > 0.5).astype(int)

    nn = NeuralNetwork([DenseLayer(2, 1, activation='sigmoid')])
    nn.train(X, y, learning_rate=0.01, epochs=100)
    predictions = nn.forward(X)
    
    y_linear = 2 * X[:, 0:1] + 3 * X[:, 1:2] + np.random.randn(100, 1) * 0.1

    nn_linear = NeuralNetwork([DenseLayer(2, 1, activation='linear')])
    nn_linear.train(X, y_linear, learning_rate=0.01, epochs=100)
    predictions_linear = nn_linear.forward(X)