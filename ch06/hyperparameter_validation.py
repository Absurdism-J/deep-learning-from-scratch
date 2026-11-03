import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
# __file__ = 当前脚本的路径，比如 project/ch06/main.py
# .resolve() = 转成绝对路径（消掉 ..、符号链接等），比如 /home/me/project/ch06/main.py
# .parent = 所在目录 → /home/me/project/ch06
# .parent.parent = 上一级 → /home/me/project
# 加进 sys.path(Python 查找模块/包时用的搜索路径列表)
# 若使用的是sys.path.append(".."),则添加相对目前终端运行目录的上一级目录进入搜索列表

from dataset.mnist import load_mnist
from ch05.two_layer_net_update import TwoLayerNetUpdate
# 在搜索路径寻找叫ch05的包,找到后在包范围内寻找引用模块

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)


# 1.打乱训练数据顺序
rng = np.random.default_rng(0)
# np.random.default_rng(0) 创建一个新的随机数生成器，用 0 作为种子初始化
indices = rng.permutation(x_train.shape[0])
# x_train.shape[0] = 训练样本数量
# rng.permutation(n) = 返回 0 到 n-1 的一个随机打乱排列，比如 [3, 0, 59999, 1, ...]，长度 n，每个数出现且仅出现一次
x_train = x_train[indices]
t_train = t_train[indices]


# 2.从训练数据中划出验证集
val_size = 5000
x_val = x_train[:val_size]
t_val = t_train[:val_size]  # 验证集

x_train = x_train[val_size:]
t_train = t_train[val_size:]  # 训练集


# 3. 每组超参数都重新创建网络，进行相同的短训练
def train_and_evaluate(lr):
    network = TwoLayerNetUpdate(input_size=784, hidden_size=50, output_size=10)
    batch_size = 100
    iterations = 500
    for i in range(iterations):
        batch_mask = np.random.choice(x_train.shape[0], batch_size, replace = False)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        grad = network.gradient_update(x_batch, t_batch)   # 求梯度
        for key in ('W1','b1','W2','b2'):  # 权重更新
            network.param[key] -= lr * grad[key]
    return network.accuracy(x_val, t_val)  # 训练后返回准确率


# 4. 在对数尺度上随机采样学习率
results = []
for trial in range(10):
    lr = 10**rng.uniform(-2,-1)
    # rng.uniform(-3, -1)：在[-3, -1) 区间均匀随机取一个浮点数
    # 直接在[0.001, 0.1]上均匀取,显然0.01~0.1区间很大,大部分值会落在0.01~0.1这段,小值很少
    # 先在对数空间[-3, -1]均匀取，再10 ** 还原，得到对数均匀分布,0.001~0.01和0.01~0.1两段被选中的概率相等
    # 对数空间是把数值取对数之后所在的那个尺度
    val_acc = train_and_evaluate(lr)
    results.append((lr, val_acc))
    print(
        f"trial={trial + 1}, "
        f"lr={lr:.6f}, "
        f"val_accuracy={val_acc:.4f}"
    )


# 5. 按验证集准确率从高到低排序
results.sort(key = lambda result: result[1], reverse = True)
# sort()：原地修改 results
# lambda的result是一个任意的变量名,接收sort遍历传来的results列表中的元素
# key：指定排序标准。这里 result 是列表中的单个元素（如元组 (lr, val_acc)），result[1]取其第2项(验证准确率)作为排序标准
# 可多标准排序，如 (result[1], result[0]),先比result[1],相同再比result[0]
# reverse=True：降序，准确率高的排前面
for lr, val_acc in results:
    # for负责遍历列表results
    # 元组解包：每次循环取一个元组，把它的第1项赋给lr，第2项赋给val_accuracy
    # 解包是"把一个容器里的元素一次性拆给多个变量"
    print(f"lr={lr:.6f}, val_accuracy={val_acc:.4f}")

"""
第一轮在 0.001～0.1 内随机搜索，较好区域集中在 0.05 附近；
第二轮缩小到 0.01～0.1 后，当前最佳学习率约为 0.07，验证集准确率为 88.54%。
该结果只代表当前搜索范围与短训练设置下的较优值，不是全局最优学习率。
"""

