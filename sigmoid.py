import numpy as np
def sigmoid(x):
    y = 1 / (1+np.exp(-x))
    return y
if __name__ == "__main__":
    a = int(input())
    x = np.array([float(input())for _ in range(a)])
    print(sigmoid(x))