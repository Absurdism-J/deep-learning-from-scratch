import numpy as np
def numerical_diff(func,x):
    h = 1e-4
    return (func(x+h)-func(x-h))/(2*h)

def func1(x):
    return 0.01*x**2 + 0.1*x

def func2(x):
    return x[0]**2 + x[1]**2

# def numerical_gradient(func,x):
#     h = 1e-4
#     G = np.zeros_like(x)
#     for i in range(x.size):
#         temp = x[i]
#         x[i] = temp+h
#         f1 = func(x)
#         x[i] = temp-h
#         f2 = func(x)
#         G[i] = (f1-f2)/(2*h)
#         x[i] = temp
#     return G

def numerical_gradient(func,x):
    h = 1e-4
    G = np.zeros_like(x,dtype=np.float64)
    # 防止x是整形,保证输出是小数时不被截断
    for i in range(x.size):
        def g(t):
            # t是形参,numerical_diff调用函数时自动赋值,两次调用t值分别为x+h和x-h
            temp = x[i]
            x[i] = t
            # 注意x数组的元素类型,否则小数赋值整形会被截断
            val = func(x)
            x[i] = temp
            return val
        G[i] = numerical_diff(g,x[i])
    return G

if __name__ == "__main__":
    x0 = 5
    x1 = np.array([3,4],dtype=np.float64)
    print(numerical_diff(func1,x0))
    print(numerical_gradient(func2,x1))

