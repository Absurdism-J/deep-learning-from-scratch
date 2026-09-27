import numpy as np
from numerical_diff import numerical_gradient
from numerical_diff import func2
def gradient_descent(func,x_init,lr=0.01,step_num=100):
    x = np.array(x_init, dtype=np.float64)
    for i in range(step_num):
        G = numerical_gradient(func,x)
        x -= lr * G
    return x

if __name__ == "__main__":
    x = np.array([-3.0, 4])
    print(gradient_descent(func2,x,lr=0.1))
