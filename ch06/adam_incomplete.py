import numpy as np
class Adam:
    def __init__(self, lr, momentum):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        self.h = None

    def update(self, param, grads):
        if self.h is None and self.v is None:
            self.h, self.v = {}, {}
            for key, val in param.items():
                self.h[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        for key in param.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            self.h[key] += grads[key] * grads[key]
            param[key] += self.v[key] / np.sqrt(self.h[key] + 1e-7)