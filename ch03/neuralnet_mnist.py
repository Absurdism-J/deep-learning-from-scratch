import sys
sys.path.append("..")
from dataset.mnist import load_mnist
def get_data():
    """加载mnist数据集,对其进行处理后,提取测试集数据"""
    pass

def init_network():
    "读取sample_weight.pkl,按对应层神经网络分别获取权重和偏置"
    pass

def predict(network, x):
    """
    接收测试集的输入,将输出通过三层神经网络,输出各个数的原始分数的数组
    三层前向传播,返回10维向量,softmax概率用于学习阶段,真实标签会在后续与预测结果进行对比
    """
    pass




