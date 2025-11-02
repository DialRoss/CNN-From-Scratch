import numpy as np

# === CLASSES DE LA SEMAINE 1 (DÉBUT) ===
class Dense:
    def __init__(self, input_dim, output_dim):
        # He initialization pour ReLU
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
        self.b = np.zeros(output_dim)
        self.grad_W = None
        self.grad_b = None
        self.input = None

    def forward(self, X):
        self.input = X
        return np.dot(X, self.W) + self.b

    def backward(self, grad_output):
        self.grad_W = np.dot(self.input.T, grad_output)
        self.grad_b = np.sum(grad_output, axis=0)
        return np.dot(grad_output, self.W.T)

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
        # Stabilité numérique
        exp = np.exp(X - np.max(X, axis=1, keepdims=True))
        self.output = exp / np.sum(exp, axis=1, keepdims=True)
        return self.output

    def backward(self, grad_output):
        # La dérivée de softmax + cross_entropy est simplifiée
        return grad_output
# === CLASSES DE LA SEMAINE 1 (FIN) ===

# === CLASSES DE LA SEMAINE 2 (DÉBUT) ===
class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # He initialization pour ReLU
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2. / (in_channels * kernel_size * kernel_size))
        self.b = np.zeros(out_channels)
        
        self.grad_W = None
        self.grad_b = None
        self.input = None

    def forward(self, X):
        batch_size, in_channels, in_height, in_width = X.shape
        
        out_height = (in_height + 2*self.padding - self.kernel_size) // self.stride + 1
        out_width = (in_width + 2*self.padding - self.kernel_size) // self.stride + 1
        
        if self.padding > 0:
            X_padded = np.pad(X, ((0,0), (0,0), (self.padding,self.padding), (self.padding,self.padding)), mode='constant')
        else:
            X_padded = X
        
        self.input = X_padded
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))
        
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                
                patch = X_padded[:, :, h_start:h_end, w_start:w_end]
                
                for k in range(self.out_channels):
                    # Broadcast multiplication + sum over channels, kernel_height, kernel_width
                    output[:, k, i, j] = np.sum(patch * self.W[k, :, :, :], axis=(1,2,3)) + self.b[k]
        
        return output

    def backward(self, grad_output):
        batch_size, _, out_height, out_width = grad_output.shape
        X_padded = self.input
        
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)
        grad_input = np.zeros_like(X_padded)
        
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                
                patch = X_padded[:, :, h_start:h_end, w_start:w_end]
                
                for k in range(self.out_channels):
                    grad_k = grad_output[:, k, i, j][:, None, None, None]
                    self.grad_W[k] += np.sum(patch * grad_k, axis=0)
                    self.grad_b[k] += np.sum(grad_output[:, k, i, j])
                    grad_input[:, :, h_start:h_end, w_start:w_end] += self.W[k] * grad_k
        
        if self.padding > 0:
            grad_input = grad_input[:, :, self.padding:-self.padding, self.padding:-self.padding]
        
        return grad_input

class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.input = None
        self.max_indices = None

    def forward(self, X):
        self.input = X
        batch_size, channels, in_height, in_width = X.shape
        
        out_height = (in_height - self.pool_size) // self.stride + 1
        out_width = (in_width - self.pool_size) // self.stride + 1
        
        output = np.zeros((batch_size, channels, out_height, out_width))
        self.max_indices = np.zeros((batch_size, channels, out_height, out_width, 2), dtype=int)
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size
                        
                        patch = X[b, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(patch)
                        output[b, c, i, j] = max_val
                        max_idx = np.unravel_index(np.argmax(patch), patch.shape)
                        self.max_indices[b, c, i, j] = max_idx
        
        return output

    def backward(self, grad_output):
        grad_input = np.zeros_like(self.input)
        batch_size, channels, out_height, out_width = grad_output.shape
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        h_idx, w_idx = self.max_indices[b, c, i, j]
                        grad_input[b, c, h_start + h_idx, w_start + w_idx] += grad_output[b, c, i, j]
        
        return grad_input

class Flatten:
    def __init__(self):
        self.input_shape = None
    
    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)
    
    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)
# === CLASSES DE LA SEMAINE 2 (FIN) ===