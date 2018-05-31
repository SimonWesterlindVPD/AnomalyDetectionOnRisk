from garch import garch, delta_to_cum
import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt

n = 10000
ω, α, β = 1, [10], [10]
y = garch(ω, α, β, n, lambda n: np.random.normal(0, 1, n))
x = range(len(y))

# plt.hist(y, 50, normed=True)
# x1 = np.arange(-10, 10, 0.1)
# y1 = sp.stats.norm.pdf(x1, 0, 1)
# plt.plot(x1, y1)
# plt.show()

plt.plot(x, y, label="garch")
# plt.plot(x, delta_to_cum(y), label="garch cum")

y_norm = np.random.normal(0, 1, n)
plt.plot(x, y_norm + 5, label="gauss")
# plt.plot(x, delta_to_cum(y_norm), label="gauss cum")

plt.legend(loc='upper right')
plt.show()
