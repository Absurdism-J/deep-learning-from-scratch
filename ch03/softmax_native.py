import numpy as np
def softmax(x):
    e_x = np.exp(x)
    s_e_x = np.sum(e_x)
    return e_x / s_e_x
if __name__ == "__main__":
    a = np.array([0.3,2.9,4.0])
    print(softmax(a))
    b = np.array([1010, 1000, 990])
    print(softmax(b))