import numpy as np
def softmax(x):
    m = np.max(x)
    e_x = np.exp(x-m)
    s_e_x = np.sum(e_x)
    return e_x / s_e_x
if __name__ == "__main__":
    a = np.array([1010,1000,990])
    print(softmax(a))