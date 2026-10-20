import numpy as np
"""
记住历史梯度,当前梯度会与历史梯度进行矢量合成,
因为梯度不总是指向谷底,所以反向会抵消会让偏移幅度减小,
同向加速,效率提高
"""
class Momentum:
    def __init__(self, lr, momentum):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, param, grads):
        if self.v is None:
            self.v = {}    # 进行梯度下降时会多次更新梯度,v会在原来的基础上多次变化,只需要初始化一次
            for key, value in param.items():
                self.v[key] = np.zeros_like(value)
        for key in grads:
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            param[key] += self.v[key]

if __name__ == "__main__":
    opt = Momentum(lr=0.1, momentum=0.9)
    param = {"x": np.array([5.0])}   # f(x)=x²，起点在谷底右侧 5 单位处
    for step in range(50):
        grads = {"x": 2 * param["x"]}
        opt.update(param, grads)
    print("final f(x) =", param["x"]**2)
