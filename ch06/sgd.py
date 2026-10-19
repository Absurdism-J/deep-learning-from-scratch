class SGD:
    def __init__(self, lr = 0.01):
        self.lr = lr

    def update(self, param, grads):
        for key in param.keys():
            param[key] -= self.lr * grads[key]

