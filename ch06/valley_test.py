import numpy as np
from sgd import SGD
from momentum import Momentum
from adagrad import Adagrad
from adam_incomplete import Adam
"""
测试地形一个维度坡度大,另一个坡度小,
坡度大的当前梯度与历史梯度反向抵消,坡度小的叠加
利好momentum方法
"""
def valley_test(opt, steps=200):
    param = {'x': np.array([5.0]), 'y': np.array([5.0])}
    for _ in range(steps):
        grads = {'x': 2 * param['x'], 'y': 100 * param['y']}
        opt.update(param, grads)
    return param['x']**2 + 50 * param['y']**2


test1 = valley_test(SGD(lr=0.01))
print(test1)
test2 = valley_test(Momentum(lr=0.01, momentum=0.9) )
print(test2)
test3 = valley_test(Adagrad(lr=0.01))
print(test3)
test4 = valley_test(Adam(lr=0.01, momentum=0.9))
print(test4)
