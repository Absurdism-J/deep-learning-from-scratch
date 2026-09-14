import numpy as np
def OR(x):
    w = np.array([1,1])
    b = -0.5
    if(np.sum(x*w)+b>0):
        return 1
    else:
        return 0
x = np.ones(2)
for i in range(2):
    x[i] = float(input())
print(OR(x))
