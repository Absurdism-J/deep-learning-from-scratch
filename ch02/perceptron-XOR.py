import numpy as np
from perceptron_one import NAND
from perceptron_two import AND
from perceptron_three import OR
a = np.ones(2)
for i in range(2):
    a[i] = int(input())
re1 = OR(a)
re2 = NAND(a)
b = np.array([re1,re2])
result = AND(b)
print(result)
