import numpy as np
def step_function(x):
    y = x > 0
    return y.astype(np.int_)
if __name__ == "__main__":
    a = int(input())
    x = np.ones(a)
    for i in range(a):
        x[i] = float(input())
    print(step_function(x))