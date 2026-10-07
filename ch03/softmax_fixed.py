import numpy as np
def softmax(x):
    # if x.ndim == 2:                    # batch：二维输入
    #     x = x.T                        # 转置，让"每张图"变成列
    #     x = x - np.max(x, axis=0)      # 每列（每张图）减自己的最大值
    #     y = np.exp(x) / np.sum(np.exp(x), axis=0)   # 每列各自归一
    #     return y.T                     # 转回去
    # x = x - np.max(x)                  # 一维：原逻辑
    # return np.exp(x) / np.sum(np.exp(x))

    c = np.max(x, axis=1, keepdims=True)  # (50,1)：每行一个 max    axis:固定其余维度,按指定维度方向执行语句
    # keepdims=True:保存被删除的维度,sum/mean(计算平均数)/max/argmax等归约函数(用一个数代表一组数的函数)会删去指定维度
    exp_x = np.exp(x - c)  # (50,1) 广播到 (50,10)
    sum_exp = np.sum(exp_x, axis=1, keepdims=True)  # (50,1)：每行一个分母
    return exp_x / sum_exp  # 广播除法

if __name__ == "__main__":
    a = np.array([1010,1000,990])
    print(softmax(a))