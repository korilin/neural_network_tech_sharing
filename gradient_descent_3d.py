from matplotlib import pyplot, cm
import numpy as np

fig = pyplot.figure()
axes = pyplot.axes(projection='3d')

xx = np.arange(-10, 10, 0.1)
yy = np.arange(-10, 10, 0.1)
X, Y = np.meshgrid(xx, yy)
Z = X ** 2 + Y ** 2 + 10

axes.plot_surface(X, Y, Z, alpha=0.9, cmap=cm.coolwarm)
axes.contour(X, Y, Z, zdir='z', offset=-5, cmap="rainbow")

axes.set_xlabel('w1')
axes.set_xlim(-9, 9)
axes.set_ylabel('w2')
axes.set_ylim(-9, 9)
axes.set_zlabel('cost')
axes.set_zlim(-5, 200)

pyplot.show()
