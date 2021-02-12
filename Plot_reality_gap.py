import pickle
import utils
from matplotlib import pyplot as plt
import numpy as np

windows_with_errors = pickle.load(open("data/reality_gap/gap.pkl" , "rb"))


ys = utils.to_array_truncate(list(map(lambda w: np.array(w["errors"]), windows_with_errors)))
print(ys.shape)
ys = ys[:, :1000]
y = np.mean(ys, axis=0)
y_std = np.std(ys, axis=0)
x = windows_with_errors[0]["t"][:len(y)]
fig, ax = plt.subplots(figsize=(5,4))
legend_stuff = utils.plot_with_std(ax, x, y, y_std)
print(y_std[1])
print(y[1])
ax.set_xlabel("Time (s)")
ax.set_ylabel("APG (degrees/s)")
ax.legend(legend_stuff, ("std", "mean"))
plt.show()