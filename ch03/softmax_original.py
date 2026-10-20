import numpy as np
def softmax(x):
    e_x = np.exp(x)
    s_e_x = np.sum(e_x)
    return e_x / s_e_x
# 若x值较大,分子分母均趋于零,0/0计算无意义,结果为nan(inf/inf也会)
# 若是数字过大溢出,结果为inf
if __name__ == "__main__":
    a = np.array([0.3,2.9,4.0])
    print(softmax(a))
    b = np.array([1010, 1000, 990])
    print(softmax(b))