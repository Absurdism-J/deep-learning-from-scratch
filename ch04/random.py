import numpy as np
def random(x,t,s):
    train_size = x.shape[0]
    batch_size = s
    batch_mask = np.random.choice(train_size,batch_size)
    x_batch = x[batch_mask]
    t_batch = t[batch_mask]
    return x_batch, t_batch