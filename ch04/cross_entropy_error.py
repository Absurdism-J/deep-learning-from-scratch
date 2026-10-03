import numpy as np

def cross_entropy_error_onehot(y,t):
# 此处y和t是作为已经筛选出的批数据mini_batch集中一次性输入
    if y.ndim == 1:
        y = y.reshape(1,y.size)
        t = t.reshape(1,t.size)
    # 二维化是为了统一除法,让图片数量正好等于第零维(第零维行数是图片个数).否则输入一维的话,batch_size就变成图片参数个数了
    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))/batch_size
    # 使用独热标签形式的表达,计算时会自然剩下正确答案的概率

def cross_entropy_error(y, t):
    # 此处y和t是作为已经筛选出的批数据mini_batch集中一次性输入
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    # 二维化是为了统一除法,让图片数量正好等于第零维元素个数.否则一维的batch_size就从图片数量1变成参数个数了
    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size
    # 不使用独热标签形式的表达,按行顺序在t中获取对应的索引(答案正好等于索引),最终取到所需要的概率

if __name__ == "__main__":
    # 情况1：一维单图（触发 reshape 分支）
    y = np.array([0.1, 0.7, 0.2])      # 预测：给1号类别7成把握
    t_onehot = np.array([0, 1, 0])     # 独热答案：就是1号
    t_label = np.array([1])                        # 整数标签版要的格式
    print(cross_entropy_error_onehot(y, t_onehot))# 应得 -log(0.7) ≈ 0.357
    print(cross_entropy_error(y, t_label))

    # 情况2：batch 两张图
    y = np.array([[0.1, 0.7, 0.2],
                  [0.8, 0.1, 0.1]])    # 第二张猜错了（正确答案1号，只给了0.1）
    t_onehot = np.array([[0, 1, 0],
                         [0, 1, 0]])
    t_label = np.array([1, 1])
    print(cross_entropy_error_onehot(y, t_onehot))   # 应得 (0.357+2.303)/2 ≈ 1.33
    print(cross_entropy_error(y, t_label))