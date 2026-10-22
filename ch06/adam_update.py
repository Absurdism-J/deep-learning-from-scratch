"""
(代码中叫做m,m（方向）变得快,近的数据参考价值远超更远的,所以beta1=0.9衰减快一点)
指数加权平均:V0=0, Bata为固定系数, θt是第t次的数据, Vt = Bata * Vt-1 + (1-Bata) * θt
通过递归,可以发现距离第t次为i时,此数据指数加权平均以Bata的i次方进行衰减
越近的数据权重越高,越有参考价值.实际中保存V0=0和Bata使用尾递归进行计算很便利
由于V0=0,初始指数加权平均会与实际数据差距较大,特别是序列短时.需要 Vt = Vt/(1-Bata**t)进行修正,补完零起步的债.
更新: param[key] += self.Vt[key]
此处原始数据θt为梯度grad,v稳定的是梯度方向的准确性.
"""
"""
(代码中叫做v,v（尺度）变得慢,样本越多，尺度估计越准,历史数据参考价值高,所以beta1=0.999衰减慢)
数值平方的指数加权平均:V0=0, Bata为固定系数, θt是第t次的数据, Vt = Bata * Vt-1 + (1-Bata) * θt ** 2
此处目的是为了约束步长,需要记录梯度的历史量级,平方是为了防止梯度的方向相反互相抵消.
越近的数据权重越高,越有参考价值.实际中保存V0=0和Bata使用尾递归进行计算很便利.
由于V0=0,初始指数加权平均会与实际数据差距较大,特别是序列短时.需要 Vt = Vt/(1-Bata**t)进行修正
更新:param[key] -= self.lr * grads[key] / (np.sqrt(self.Vt[key]) + 1e-7) -> RMSProp算法
指数加权平均越大,近期梯度量级越大,正常来说移动越多越接近谷底,为了避免越界需要进行约束.
/ (np.sqrt(self.Vt[key]) + 1e-7):移动越多权值越小,1e-7防止历史均值为零.
此处原始数据θt为梯度grad,v稳定的是实际步长的大小.
"""

import numpy as np
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.times = 0
        self.m = None
        self.v = None

    def update(self, param, grads):
        self.times += 1
        if self.m is None:
            self.m = {}
            self.v = {}
            for key, val in param.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        for key in grads:
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key] ** 2
            m_c = self.m[key] / (1 - self.beta1 ** self.times)
            v_c = self.v[key] / (1 - self.beta2 ** self.times)
            param[key] -= self.lr * m_c / (np.sqrt(v_c) + 1e-7)