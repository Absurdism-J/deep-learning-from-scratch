import numpy as np
import sys
sys.path.append('..')
from ch03.sigmoid import sigmoid
from ch03.softmax_fixed import softmax
from cross_entropy_error import cross_entropy_error_onehot
from numerical_diff import numerical_gradient

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std: object = 0.01) -> None:
        self.param = {}
        self.param['W1'] = weight_init_std*np.random.randn(input_size,hidden_size)
        self.param['b1'] = np.zeros(hidden_size)
        self.param['W2'] = weight_init_std*np.random.randn(hidden_size,output_size)
        self.param['b2'] = np.zeros(output_size)

    def predict(self,x):
        W1 , W2 = self.param['W1'],self.param['W2']
        b1 , b2 = self.param['b1'],self.param['b2']
        a1 = np.dot(x,W1)+b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,W2)+b2
        y = softmax(a2)
        return y

    def loss(self,x,t):
        y = self.predict(x)
        return cross_entropy_error_onehot(y,t)

    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis =1)
        t = np.argmax(t,axis =1)
        result_accuracy = np.sum(y==t)/float(y.shape[0])
        return result_accuracy

    def gradient(self,x,t):
        loss_W = lambda W: self.loss(x,t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W,self.param['W1'])
        grads['b1'] = numerical_gradient(loss_W,self.param['b1'])
        grads['W2'] = numerical_gradient(loss_W,self.param['W2'])
        grads['b2'] = numerical_gradient(loss_W,self.param['b2'])
        return grads