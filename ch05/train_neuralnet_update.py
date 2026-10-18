import numpy as np
import sys
sys.path.append("..")
from dataset.mnist import load_mnist
from two_layer_net_update import TwoLayerNetUpdate

if __name__ == '__main__':
    (x_train,t_train),(x_test,t_test) = load_mnist(normalize=True,one_hot_label=True)

    # 超参数
    iters_num = 10000    # 一共跑多少批
    train_size = x_train.shape[0]  # 训练集图片总数
    batch_size = 100  # 一批图片数量
    learning_rate = 0.1  # 学习率
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size/batch_size,1)

    network = TwoLayerNetUpdate(input_size=784,hidden_size=50,output_size=10)
    for i in range(iters_num):
        # 获取mini_batch
        batch_mask = np.random.choice(train_size,batch_size,replace=False)
        # 从train_size中随机选择batch_size张照片,batch_mask为所选图片编号对应数据集第零维
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 计算梯度
        grad = network.gradient_update(x_batch,t_batch)

        # 更新参数
        for key in ('W1','b1','W2','b2'):
            network.param[key] -= learning_rate * grad[key]   # 更新的是网络的参数

        # 记录学习过程
        if i % iter_per_epoch == 0:
            loss = network.loss(x_batch, t_batch)
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_loss_list.append(loss)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(f"iter {i / iter_per_epoch}, {loss:.4f}, {train_acc:.4f}, {test_acc:.4f}")