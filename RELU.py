import numpy as np
def relu(x):
    return np.maximum(0,x)
if __name__ == "__main__":
    # x = np.array(list(map(float,input().split())))
    x = np.array(input().split(),dtype = float)
    print(relu(x))