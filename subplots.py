import matplotlib.pyplot as plt
import numpy as np

# creating grid for subplots
fig = plt.figure()

ax1 = plt.subplot2grid(shape=(2, 2), loc=(0, 0))
ax2 = plt.subplot2grid(shape=(2, 2), loc=(0, 1))
ax3 = plt.subplot2grid(shape=(2, 2), loc=(1, 0), rowspan=1, colspan=2)

# initializing x,y axis value
x = np.arange(0, 10, 0.1)
y = np.cos(x)

# plotting subplots
ax1.plot(x, y)
ax1.set_title('ax1')
ax2.plot(x, y)
ax2.set_title('ax2')
ax3.plot(x, y)
ax3.set_title('ax3')

# automatically adjust padding horizontally
# as well as vertically.
plt.tight_layout()

# display plot
plt.show()