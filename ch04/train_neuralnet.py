import numpy as np
import sys
sys.path.append("..")
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

if __name__ == '__main__':
    (x_train,t_train),(x_test,t_test) = load_mnist(normalize=True,one_hot_label=True)
    train_loss_list = []

    # 超参数
    iters_num = 100    # 一共跑多少批
    train_size = x_train.shape[0]
    batch_size = 50
    learning_rate = 0.1


    network = TwoLayerNet(input_size=784,hidden_size=50,output_size=10)
    for i in range(iters_num):
        # 获取mini_batch
        batch_mask = np.random.choice(train_size,batch_size,replace=False)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 计算梯度
        grad = network.gradient(x_batch,t_batch)

        # 更新参数
        for key in ('W1','b1','W2','b2'):
            network.param[key] -= learning_rate * grad[key]   # 更新的是网络的参数

        # 记录学习过程
        loss = network.loss(x_batch,t_batch)
        train_loss_list.append(loss)
        if i % 10 == 0:
            print(f"iter {i}/{iters_num}, {loss:.4f}")