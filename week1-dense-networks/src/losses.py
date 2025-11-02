import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.pred = None
        self.target = None

    def forward(self, pred, target):
        """
        pred: sortie softmax (batch_size, n_classes)
        target: labels one-hot (batch_size, n_classes)
        """
        self.pred = pred
        self.target = target
        # éviter log(0) par ajout d'une petite constante eps
        eps = 1e-9
        loss = -np.sum(target * np.log(pred + eps)) / pred.shape[0]
        return loss

    def backward(self):
        # dérivée directe pour combinaison softmax + cross entropy
        grad = (self.pred - self.target) / self.pred.shape[0]
        return grad
