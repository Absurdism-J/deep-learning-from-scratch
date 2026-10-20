import numpy as np
class Adagrad:
    def __init__(self, lr):
        self.lr = lr
        self.h = None

    def update(self, param, grads):
        if self.h is None:
            self.h = {}
            for key, val in param.items():
                self.h[key] = np.zeros_like(val)
        for key in grads:
            self.h[key] += grads[key] * grads[key]
            param[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
            # + 1e-7:避免h局部分量为零,结果为NaN无意义,或者h局部分量极小,结果为inf无穷大

"""
根据基本不等式进行推导,易知均值小于等于均方根,将均方根近似看作均值,
由于lr/根号t大小每一步固定,可以通过比较当前梯度与均方根的比值,看出当前梯度比起整个历史整体上的平均梯度,陡了还是缓了.
不过缺点很明显,无法比较当前梯度与前一个梯度是陡了还是缓了.
均方根比起均值可以消除正负同时扩大极端例子的影响,RMS是均方根的意思.
"""

if __name__ == '__main__':
    param = {'X': np.array([100.0])}
    optimizer = Adagrad(lr=10000)
    # 一维只有一个谷底,梯度永远指向正确方向(方向对);且Adagrad调整了步长,前面越位也可以拉回谷底(步长正确)
    for step in range(100):
        grads = {'X': 2 * param['X']}
        optimizer.update(param, grads)
    print("final f(x) =", param['X'])

