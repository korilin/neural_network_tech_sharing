import matplotlib.pyplot as plt
import numpy as np

tanh = lambda z: (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

start = -10
stop = 10
step = 0.01
num = (stop - start) / step
x = np.linspace(start, stop, int(num))
y = tanh(x)

plt.plot(x, y, label='Tanh')
plt.grid(True)

plt.legend()
plt.show()
