import numpy as np
import sys
sys.path.append("..")
from ch03.softmax_fixed import softmax
from ch04.cross_entropy_error import cross_entropy_error_onehot


class SoftmaxWithLoss:
    def __init__(self):
        self.y = None
        self.t = None
        self.loss = None

    def forward(self, x, t):
        self.y = softmax(x*0.01)
        self.t = t
        self.loss = cross_entropy_error_onehot(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.y.shape[0]
        dx = dout * (self.y  - self.t) / batch_size
        return dx

if __name__ == '__main__':
    net = SoftmaxWithLoss()
    x = np.random.randn(10, 10)  # 10 张图、10 个类别，随机分数
    t = np.eye(10)  # 10×10 单位矩阵,独热标签
    print(net.forward(x, t))  # 应该 ≈ 2.3 附近（ln 10 随机基线！）