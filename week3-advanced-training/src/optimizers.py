import numpy as np

class SGD:
    def __init__(self, lr=0.01, momentum=0.9, weight_decay=0.0):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = {}
    
    def step(self, params):
        for idx, param in enumerate(params):
            if idx not in self.velocity:
                self.velocity[idx] = np.zeros_like(param['value'])
            
            grad = param['grad'] + self.weight_decay * param['value']
            self.velocity[idx] = self.momentum * self.velocity[idx] + grad
            param['value'] -= self.lr * self.velocity[idx]


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0
    
    def step(self, params):
        self.t += 1
        for idx, param in enumerate(params):
            if idx not in self.m:
                self.m[idx] = np.zeros_like(param['value'])
                self.v[idx] = np.zeros_like(param['value'])
            
            grad = param['grad'] + self.weight_decay * param['value']
            
            self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * grad
            self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (grad ** 2)
            
            m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
            v_hat = self.v[idx] / (1 - self.beta2 ** self.t)
            
            param['value'] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class RMSprop:
    def __init__(self, lr=0.001, beta=0.9, eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.weight_decay = weight_decay
        self.v = {}

    def step(self, params):
        for idx, param in enumerate(params):
            if idx not in self.v:
                self.v[idx] = np.zeros_like(param['value'])

            grad = param['grad']
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param['value']

            self.v[idx] = self.beta * self.v[idx] + (1 - self.beta) * (grad ** 2)
            param['value'] -= self.lr * grad / (np.sqrt(self.v[idx]) + self.eps)


class LearningRateScheduler:
    def __init__(self, initial_lr, decay_rate=0.5, decay_steps=10):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.current_lr = initial_lr
        
    def step(self, epoch):
        if epoch > 0 and epoch % self.decay_steps == 0:
            self.current_lr *= self.decay_rate
            print(f"Learning rate decayed to {self.current_lr:.6f}")
        return self.current_lr