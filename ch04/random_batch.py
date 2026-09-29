import numpy as np

def get_random_batch(x, t, batch_size, replace=False):
    """
    从 x, t 中随机抽取一个 mini-batch。
    replace=False 表示无放回抽样（每个样本最多出现一次）。
    """
    train_size = x.shape[0]
    batch_mask = np.random.choice(train_size, batch_size, replace=replace)
    x_batch = x[batch_mask]
    t_batch = t[batch_mask]
    return x_batch, t_batch