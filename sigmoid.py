import matplotlib.pyplot as plt
import numpy as np

sigmoid = lambda z: 1 / (1 + np.exp(-z))
start = -10
stop = 10
step = 0.01
num = (stop - start) / step
x = np.linspace(start, stop, int(num))
y = sigmoid(x)

plt.plot(x, y, label='Sigmoid')
plt.grid(True)

plt.legend()
plt.show()
