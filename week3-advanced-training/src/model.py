class Sequential:
    def __init__(self, layers):
        self.layers = layers
    
    def forward(self, X, training=True):
        for layer in self.layers:
            if hasattr(layer, 'training'):
                X = layer.forward(X, training)
            else:
                X = layer.forward(X)
        return X
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def get_params_and_grads(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'W'):
                params.append({'value': layer.W, 'grad': layer.grad_W})
            if hasattr(layer, 'b'):
                params.append({'value': layer.b, 'grad': layer.grad_b})
        return params
    
    def zero_grad(self):
        for layer in self.layers:
            if hasattr(layer, 'grad_W') and layer.grad_W is not None:
                layer.grad_W.fill(0)
            if hasattr(layer, 'grad_b') and layer.grad_b is not None:
                layer.grad_b.fill(0)
    
    def predict(self, X):
        logits = self.forward(X, training=False)
        return logits.argmax(axis=1)