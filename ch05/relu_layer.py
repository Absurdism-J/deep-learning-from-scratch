class ReLu():
    def __init__(self):
        self.mask = None
        # 需要什么就初始化什么属性，forward 算账顺手记账，backward 翻账簿。
        # 属性为反向传播服务

    def forward(self, x):
        self.mask = (x <= 0)
        # self.mask是一个布尔数组。x <= 0的位置为True。
        out = x.copy()
        # 直接out = x，out和x指向同一块内存，修改out会改掉原始的x
        out[self.mask] = 0
        # out[self.mask]选出所有mask为True的位置（x <= 0的位置）,将其赋值为0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx
