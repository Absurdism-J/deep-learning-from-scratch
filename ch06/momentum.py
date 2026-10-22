import numpy as np
"""
记住历史梯度,当前梯度会与历史梯度进行矢量合成,
因为梯度不总是指向谷底,所以反向会抵消会让偏移幅度减小,
同向加速,效率提高
"""
"""
指数加权平均:V0=0, Bata为固定系数, θt是第t次的数据, Vt = Bata * Vt-1 + (1-Bata) * θt
通过递归,可以发现距离第t次为i时,此数据在指数加权里面平均以Bata的i次方进行衰减
越近的数据权重越高,越有参考价值.实际中保存V0=0和Bata使用尾递归进行计算很便利
由于V0=0,初始指数加权平均会与实际数据差距较大,特别是序列短时.需要 Vt = Vt/(1-Bata**t)进行修正
更新: param[key] += self.Vt[key]
此处原始数据θt为梯度grad,v稳定的是梯度方向的准确性
"""
"""
指数加权平均将-lr * grad看作θt,即原始数据.鱼书则将grad看作θt,momentum就是Bata.
二者通过递归都可以发现:历史数据随着按Bata进行指数衰减
不同处在于最后更新时的实际学习率是(1-Bata)倍的关系,调参可以解决,没有本质区别.
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
