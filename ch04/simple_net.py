import sys
sys.path.append("..")
from Cross_Entropy_Error import cross_entropy_error_onehot
from numerical_diff import numerical_gradient
from ch03.softmax_fixed import softmax
import numpy as np
class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)
    def predict(self,x):
        return np.dot(x,self.W)
    def loss(self,x,t):
        z = self.predict(x)
        y = softmax(z)
        return cross_entropy_error_onehot(y,t)

net = simpleNet()
x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])   # 标准答案
loss1 = net.loss(x,t)
G = numerical_gradient(net.loss,x)
print(G)
