from mul_layer import MulLayer
from add_layer import AddLayer

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

apple_layer = MulLayer()
orange_layer = MulLayer()
apple_orange_layer = AddLayer()
tax_layer = MulLayer()
# layer

apple_price = apple_layer.forward(apple, apple_num)
orange_price = orange_layer.forward(orange, orange_num)
apple_orange_price = apple_orange_layer.forward(apple_price, orange_price)
tax_price = tax_layer.forward(apple_orange_price, tax)
# forward

dapple_orange_price, dtax = tax_layer.backward(1)
dapple_price, dorange_price = apple_orange_layer.backward(dapple_orange_price)
dapple, dapple_num = apple_layer.backward(dapple_price)
dorange, dorange_num = orange_layer.backward(dorange_price)
print(dapple, dapple_num, dorange, dorange_num, dtax)
# backward