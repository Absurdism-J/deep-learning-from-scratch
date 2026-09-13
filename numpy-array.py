import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0,9,0.1)
y1 = np.sin(x)
y2 = np.cos(x)
plt.plot(x, y1,label="sinx")
plt.plot(x, y2,linestyle = '--',label="cosx")
plt.xlabel('t')
plt.ylabel('v')
plt.title('sin & cos')
plt.legend()
plt.show()

