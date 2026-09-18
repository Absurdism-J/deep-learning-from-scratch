import numpy as np
from sigmoid import sigmoid
def identity_function(x):
    return x
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])
if X.shape[-1] == W1.shape[0]:
    A1 = np.dot(X, W1) + B1
else:
    raise ValueError("Incorrect shape")
Z1 = sigmoid(A1)
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])
if Z1.shape[-1] == W2.shape[0]:
    A2 = Z1.dot(W2) + B2
else:
    raise ValueError("Incorrect shape")
Z2 = sigmoid(A2)
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])
if Z2.shape[-1] == W3.shape[0]:
    A3 = np.dot(Z2, W3) + B3
else:
    raise ValueError("Incorrect shape")
Y = identity_function(A3)
print(Y)

