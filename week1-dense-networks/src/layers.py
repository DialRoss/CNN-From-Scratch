import numpy as np

class Dense:
    def __init__(self, input_dim, output_dim):
        # He initialization pour ReLU
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
        self.b = np.zeros(output_dim)
        self.grad_W = None
        self.grad_b = None
        self.input = None

    def forward(self, X):
        self.input = X  # pour backward
        return X.dot(self.W) + self.b

    def backward(self, grad_output):
        # gradient sur poids
        self.grad_W = self.input.T.dot(grad_output)
        # gradient sur biais
        self.grad_b = grad_output.sum(axis=0)
        # gradient sur entrée pour couches précédentes
        return grad_output.dot(self.W.T)

class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, X):
        self.input = X
        return np.maximum(0, X)

    def backward(self, grad_output):
        grad_input = grad_output.copy()
        grad_input[self.input <= 0] = 0
        return grad_input

class Softmax:
    def __init__(self):
        self.output = None

    def forward(self, X):
        # Stabilité numérique grâce à soustraction max par ligne
        X_shifted = X - np.max(X, axis=1, keepdims=True)
        exp_scores = np.exp(X_shifted)
        self.output = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.output

    def backward(self, grad_output):
        # Cette backward est correcte lors de la combinaison softmax + cross-entropy en loss
        return grad_output
