import math

class EarlyStoppingCallback:

    def __init__(self, patience):
        self.patience = patience
        self.best = 1e5
        self.progress = 0
        
    def step(self, current_loss):
        is_learning  = current_loss<self.best
        
        if not is_learning:
            self.progress += 1

    def should_stop(self):
        return self.progress > self.patience 

