import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# x = [128,64,32,16,8,4]
x = np.hstack(([0], 0.5 + np.array(range(6))))
labels = ["256x128", "128x128", "64x64", "32x32", "16x16", "8x8", "4x4"]
fig, ax = plt.subplots()

loop_rates = np.array([260, 400, 730, 850, 925, 963, 967])
loop_rates_interp = np.array([260, 400, 730, 850, 925, 963, 967, 1000])
sizes_interp = np.array([151, 80, 40, 20, 10, 4, 3, 0])
def test(x):
    x[x < loop_rates_interp.min()] = loop_rates_interp.min()
    x[x > loop_rates_interp.max()] = loop_rates_interp.max()
    return interp1d(loop_rates_interp, sizes_interp, kind='cubic')(x)

def test_inv(x):
    x[x < sizes_interp.min()] = sizes_interp.min()
    x[x > sizes_interp.max()] = sizes_interp.max()
    return interp1d(sizes_interp, loop_rates_interp, kind='cubic')(x)

ax2 = ax.secondary_yaxis('right', functions=(test, test_inv))
ax2.set_ylabel("size (Kb)")
ax.plot(x, loop_rates, linewidth=3, marker='o', markersize=10)
ax.grid()
ax.set_xlabel("Feedforward neural network size")
ax.set_ylabel("Loop rate (Hz)")
ax.set_ylim([240,1000])
plt.xticks(x, labels, rotation = 45)
plt.show()