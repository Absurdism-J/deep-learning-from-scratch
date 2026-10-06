class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self,x,y):
        self.x = x
        self.y = y
        out = x*y
        return out

    def backward(self,dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx,dy

if __name__  == "__main__":
    apple_price = 100
    apple_num = 2
    tax = 1.1
    M1 = MulLayer()
    M2 = MulLayer()
    # M2.x = M1.forward(apple_price,apple_num)   # forward方法调用后会自动存储x值和y值

    sum_price = M2.forward(M1.forward(apple_price,apple_num),tax)
    print(f"前向传播结果:{sum_price}")   # 前向传播

    dsum_price = 1
    dprice,dtax = M2.backward(dsum_price)
    dapple_price,dapple_num = M1.backward(dprice)
    print(dapple_price,dapple_num,dtax)  # 反向传播
