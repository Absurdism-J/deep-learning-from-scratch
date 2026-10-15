import sys
sys.path.append('..')
from ch04.numerical_diff import numerical_gradient
from relu_layer import *
from affine_batch import *
from softmax_with_loss import *
from collections import OrderedDict

class TwoLayerNetUpdate:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.param = {}
        self.param['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.param['b1'] = np.zeros(hidden_size)
        self.param['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.param['b2'] = np.zeros(output_size)
        # 初始化权重

        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.param['W1'], self.param['b1'])
        self.layers["ReLu"] = ReLu()
        self.layers["Affine2"] = Affine(self.param['W2'], self.param['b2'])
        # 生成层

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        if t.ndim != 1:
            t = np.argmax(t, axis = 1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self,x,t):
        loss_W = lambda W: self.loss(x,t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W,self.param['W1'])
        grads['b1'] = numerical_gradient(loss_W,self.param['b1'])
        grads['W2'] = numerical_gradient(loss_W,self.param['W2'])
        grads['b2'] = numerical_gradient(loss_W,self.param['b2'])
        return grads

    def gradient_update(self, x, t):
        self.loss(x, t)
        # 前向传播的y和t已经记录在last_layer这个类的成员变量里,在反向传播时需要被用到
        # 不需要用到返回值
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(reversed(self.layers.values()))
        for layer in layers:
            dout = layer.backward(dout) # 完整反向传播,各个梯度均被记录在对应层的类成员变量中

        grads = {}
        grads["W1"] = self.layers["Affine1"].dW
        grads["b1"] = self.layers["Affine1"].db
        grads["W2"] = self.layers["Affine2"].dW
        grads["b2"] = self.layers["Affine2"].db

        return grads

if __name__ == '__main__':
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    g = numerical_gradient(lambda w: np.sum(w ** 2), x)
    print(g)
