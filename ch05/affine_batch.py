import numpy as np


class Affine():
    def __init__(self, W, b):
        self.W = W
        self.b = b
        # W,b是神经网络通用数值,不随批次直接改变,而是因梯度进行调整
        self.x = None
        self.dW = None
        self.db = None
        # 流水记录：随批次更新只用一次

    def forward(self, x):
        self.x = x  # x每批的流水记录,行业默认在forward随每批的数据更新
        out = np.dot(self.x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        # 矩阵x*W意味着每个W或x通过乘法影响多个对应的x或W,此处反向传播时矩阵正好加上了所有影响量
        # 画矩阵图理解更清晰,下面的b自然同理,一个神经元对应一个b,一个b影响一个神经元内多个分量,影响量也要相加
        self.db = np.sum(dout, axis=0)
        # dout 一列对应一个神经元（类别）；axis=0 沿样本轴汇总——b 被 N 个样本的类别分量共用，收 N 份账
        return dx
