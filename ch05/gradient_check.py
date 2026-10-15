import sys
import numpy as np
sys.path.append('..')
from two_layer_net_update import TwoLayerNetUpdate
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNetUpdate(input_size=784, hidden_size=50, output_size=10)
grad_backprop = network.gradient_update(x_train[:4], t_train[:4])        # 反向版
grad_numerical = network.gradient(x_train[:4], t_train[:4])   # 数值版（慢，耐心等）


for key in grad_numerical.keys():
    diff = np.average(abs(grad_numerical[key] - grad_backprop[key]))
    print(f"key: {key}, diff: {diff}")

# 工程纠错:
# for key in grad_numerical.keys():
#     ratio = np.average(abs(grad_numerical[key])) / np.average(abs(grad_backprop[key]))
#     print(key, ratio)
#
# print(network.loss(x_train[:4], t_train[:4]))

# h = 1e-4
# xb, tb = x_train[:4], t_train[:4]
#
# net = TwoLayerNetUpdate(784, 50, 10)
# net.param['b2'][0] += h
# f1 = net.loss(xb, tb)
# net.param['b2'][0] -= 2*h
# f2 = net.loss(xb, tb)
# net.param['b2'][0] += h   # 恢复原值
#
# manual = (f1 - f2) / (2*h)
# backprop = net.gradient_update(xb, tb)['b2'][0]
# print("手工数值:", manual)
# print("反向传播:", backprop)
# print(id(net.param['W1']), id(net.layers['Affine1'].W))
# print(id(net.param['b2']), id(net.layers['Affine2'].b))
#
# for k in ('W1','b1','W2','b2'):
#     print(k, grad_numerical[k].shape, grad_backprop[k].shape)