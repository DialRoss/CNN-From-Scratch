import numpy as np
from src.layers import Dense

def numerical_gradient(f, x, eps=1e-6):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index

        old_val = x[idx]

        x[idx] = old_val + eps
        fx_plus = f(x)

        x[idx] = old_val - eps
        fx_minus = f(x)

        grad[idx] = (fx_plus - fx_minus) / (2 * eps)

        x[idx] = old_val
        it.iternext()
    return grad

def test_dense_gradient():
    np.random.seed(0)
    layer = Dense(3, 2)
    X = np.random.randn(4, 3)

    # Forward pass function for numerical gradient w.r.t W
    def forward_W(W_flat):
        W = W_flat.reshape(layer.W.shape)
        layer.W = W
        out = layer.forward(X)
        return out.sum()  # Scalar output for gradient check

    # Numerical gradient check for W
    grad_numeric_W = numerical_gradient(forward_W, layer.W.flatten())
    layer.forward(X)
    grad_output = np.ones_like(layer.forward(X))  # gradient of sum wrt output is ones
    layer.backward(grad_output)
    grad_analytic_W = layer.grad_W.flatten()

    print("Numerical gradient W vs Analytical gradient W:")
    print(np.allclose(grad_numeric_W, grad_analytic_W, rtol=1e-5))

if __name__ == "__main__":
    test_dense_gradient()
