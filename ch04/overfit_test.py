import sys
sys.path.append("..")
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train,t_train),(x_test,t_test) = load_mnist(normalize=True,flatten=True,one_hot_label=True)
x_small = x_train[:10]
t_small = t_train[:10]

network = TwoLayerNet(input_size=784,hidden_size=10,output_size=10)

iters_num = 50
learning_rate = 0.5
loss_list = []

for i in range(iters_num):
    grad = network.gradient(x_small,t_small)
    for key in ('W1','b1','W2','b2'):
        network.param[key] -= learning_rate * grad[key]
    print(f"W1总和={network.param['W1'].sum():.4f}  梯度W1总和={grad['W1'].sum():.6f}")  # ← 探针
    loss = network.loss(x_small,t_small)
    loss_list.append(loss)
    if i % 5 == 0:
        print(f"iter:{i:2d},loss:{loss:.4f}")
print(f"最终 loss = {loss_list[-1]:.4f}")
print("判定：< 1.0 机制健康；纹丝不动则机制有鬼")
