import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 50)
y = 2 * x + 1
z = x**2

# plt.figure(num = 1)
# plt.plot(x, y)

plt.figure(num = 3, figsize = (8, 5))

# plt.xlim(-1, 2)
# plt.ylim(-1, 2)

plt.subplot(111)

plt.xlabel('n or m')
plt.ylabel('b or g')


plt.xticks([-2, -0.5, 0, 0.5, 2], ['nn', 'n', 'o', 'm', 'mm'])
plt.yticks([-2, -0.5, 0, 0.5, 2], ['bb', 'b', 'o', 'g', 'gg'])

plt.plot(x, y, label='line')
plt.plot(x, z, label='unline')
plt.legend()


x0 = 1
y0 = 2 * x0 + 1
plt.scatter(x0, y0, s = 50, color = 'b')
plt.plot([x0, x0], [y0, 0], 'k--', lw = 2.5)

ax = plt.gca()

ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_position(('data', 0))

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

for label in ax.get_xticklabels() + ax.get_yticklabels():
  label.set_fontsize(12)
  label.set_bbox(dict(facecolor = 'white', edgecolor = 'None', alpha = .7))

# plt.grid(True)

n = 1024
x = np.random.normal(0, 1, n)
y = np.random.normal(0, 1, n)
t = np.arctan2(y, x)

plt.scatter(x, y, s = 75, c = t, alpha = .5)

plt.show()
