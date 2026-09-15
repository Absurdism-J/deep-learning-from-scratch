import numpy as np
def OR(x):
    w = np.array([0.5,0.5])
    b = -0.2
    if(np.sum(x*w)+b>0):
        return 1
    else:
        return 0
if __name__ == "__main__":
    x_test = np.ones(2)
    for i in range(2):
        x_test[i] = float(input())
    print(OR(x_test))
