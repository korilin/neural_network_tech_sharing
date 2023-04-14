import matplotlib.pyplot as plt
import numpy as np

rule = lambda z: np.maximum(0, z)

start = -10
stop = 10
step = 0.1
num = (stop - start) / step
x = np.linspace(start, stop, int(num))
y = rule(x)

plt.plot(x, y, label='ReLU')
plt.grid(True)

plt.legend()
plt.show()
