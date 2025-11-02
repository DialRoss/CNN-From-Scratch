import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.pred = None
        self.target = None
    
    def forward(self, pred, target):
        self.pred = pred
        self.target = target
        pred_clipped = np.clip(pred, 1e-12, 1-1e-12)
        loss = -np.sum(target * np.log(pred_clipped)) / pred.shape[0]
        return loss
    
    def backward(self):
        grad = (self.pred - self.target) / self.pred.shape[0]
        return grad