import numpy as np

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
    
    def on_epoch_end(self, epoch, val_loss, model=None):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                return True
        return False
    
class LearningRateScheduler:
    def __init__(self, initial_lr, decay_rate=0.5, decay_steps=10):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

    def step(self, epoch):
        # diminue le learning rate selon un facteur tous les 'decay_steps' epochs
        factor = self.decay_rate ** (epoch // self.decay_steps)
        return self.initial_lr * factor
