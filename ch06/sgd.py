"""
图像,函数,高度线与梯度都是连续的,
相邻两条高度线实际上可以认为形状完全一样”，这在局部、极小范围的意义上是合理的，因为梯度是极小范围内的偏导数。
此外,再补上两点线段最短,相邻两条高度线实平行,
且变化最小但是下降最快的方向就是负梯度的方向,所以梯度必然垂直等高线
"""
class SGD:
    def __init__(self, lr = 0.01):
        self.lr = lr

    def update(self, param, grads):
        for key in param.keys():
            param[key] -= self.lr * grads[key]

