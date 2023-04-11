import numpy as np
import activations

class DenseLayer:
    def __init__(self, n_batches, n_features, activation):
        '''
        This line of code initializes the weights using a technique called "He initialization" 
        (also known as "He normal initialization"), which is named after Kaiming He, 
        who proposed it in the 2015 paper 
        Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
        '''
        self.weights = np.random.randn(n_batches, n_features) * np.sqrt(2.0 / n_batches)
        self.bias = np.zeros((1, n_batches))
        self.activation = activation
        self.inputs = None
        
    def forward(self, inputs):
        self.inputs = inputs
        # Using the weights and bias calculated using gradient descent calculate the output 
        # for one layer as input for the next layer
        if self.activation == 'linear':
            return activations.linear(inputs, self.weights, self.bias)
        elif self.activation == 'sigmoid':
            return activations.sigmoid(inputs, self.weights, self.bias)
        elif self.activation == 'relu':
            return activations.relu(inputs, self.weights, self.bias)
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
        for layer in self.layers:
            if layer.activation == 'linear':
                layer.weights,layer.bias = activations.train_linear_regression(X, y, learning_rate, epochs)
            elif layer.activation == 'sigmoid':
                layer.weights, layer.bias = activations.train_logistic_regression(X, y, learning_rate, epochs)
            elif layer.activation == 'relu':
                layer.weights, layer.bias = activations.train_relu(X, y, learning_rate, epochs)