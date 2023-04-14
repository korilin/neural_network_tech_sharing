from matplotlib import pyplot
import numpy as np

fig = pyplot.figure()

X = np.arange(-10, 10, 0.1)
Y = X ** 2

pyplot.plot(X, Y)
pyplot.xlabel("w")
pyplot.ylabel("cost")
pyplot.ylim(0, 100)
pyplot.ylim(0, 100)
pyplot.xticks([])
pyplot.yticks([])

pyplot.show()
