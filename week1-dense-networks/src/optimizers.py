import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params):
        for param in params:
            param['value'] -= self.lr * param['grad']

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, params):
        self.t += 1
        for idx, param in enumerate(params):
            if idx not in self.m:
                self.m[idx] = np.zeros_like(param['value'])
                self.v[idx] = np.zeros_like(param['value'])
            self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * param['grad']
            self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (param['grad'] ** 2)
            m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
            v_hat = self.v[idx] / (1 - self.beta2 ** self.t)
            param['value'] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
