import numpy as np
class Dropout:
    def __init__(self, drop_ratio):
        self.drop_ratio = drop_ratio
        self.mask = None

    def forward(self, x, train_flag = True):
        if train_flag:    # 学习/训练时
            self.mask = np.random.rand(*x.shape) > self.drop_ratio
            # np.random.rand生成的是[0, 1) 区间上的均匀分布随机数，左闭右开：可能取到 0，但取不到 1。
            # *x.shape代表生成形状和x相同的矩阵.
            # np.random.rand(*x.shape) > self.drop_ratio,逐个比大小生成布尔矩阵,大于为True
            return x * self.mask / (1.0 - self.drop_ratio)
        else:
            return x

    def backward(self, dout):
        return dout * self.mask / (1.0 - self.drop_ratio)
        # 训练均值尺度缩放了1/(1.0 - self.drop_ratio),反向传播求梯度要乘回去

if __name__ == '__main__':
    # 验证 forward 和 backward的缩放系数是否一致

    np.random.seed(1)
    # 固定随机种子,使每次生成的layer.mask相同

    layer = Dropout(0.5)
    x = np.array([1.0, 2.0, 3.0, 4.0])

    y = layer.forward(x, train_flag=True)
    print("mask:", layer.mask)
    print("y:", y)

    dx = layer.backward(np.ones_like(x))
    print("dx:", dx)

    # 除以0.5,y与dx均被放大两倍,缩放系数一致