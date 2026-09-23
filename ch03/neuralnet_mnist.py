import sys, pickle
import numpy as np
from sigmoid import sigmoid
from softmax_fixed import softmax
sys.path.append("..")
from dataset.mnist import load_mnist
def get_data():
    (x_train,t_train), (x_test, t_test) = load_mnist(normalize = True, flatten = True, one_hot_label = False)
    return x_test, t_test
    """加载mnist数据集,对其进行处理后,提取测试集数据"""


def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
        return network
    "读取sample_weight.pkl,按对应层神经网络分别获取权重和偏置"


def predict(network, x):
    W1,W2,W3 = network["W1"],network["W2"],network["W3"]
    b1,b2,b3 = network["b1"],network["b2"],network["b3"]
    a1 = np.dot(x,W1)+b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2)+b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3)+b3
    return a3
    # y = softmax(a3)
    # return y

    # 接收测试集的输入,将输出通过三层神经网络,输出各个数的原始分数的数组
    # 三层前向传播,返回10维向量,softmax概率用于学习阶段,真实标签会在后续与预测结果进行对比

x, t = get_data()
network = init_network()
accuracy = 0
for j in range(len(x)):
    data = predict(network, x[j])
    # "直接把每一行的元素(相当于一张图)当作一个数组取出进行运算"
    p = np.argmax(data)
    # "此处索引正好对应作为输出层0到9"
    if p == t[j]:
        accuracy += 1
print(f"Accuracy: {accuracy/len(x)*100}%")









