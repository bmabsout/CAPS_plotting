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
t = windows_with_errors[0]["t"][:len(y)]
dt = t[1:] - t[:-1]
ys_d = (ys[:, 1:] - ys[:, :-1])/dt
print(np.shape(ys_d))
print("change in rate", np.mean(ys_d))
fig, ax = plt.subplots(2, figsize=(5,4))
legend_stuff = utils.plot_with_std(ax[0], t, y, y_std)
print(y_std[1])
print(y[1])
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("APG (degrees/s)")
ax[0].legend(legend_stuff, ("std", "mean"))
ax[0].set_yscale('symlog')
ax[1].set_yscale('symlog')
utils.plot_with_std(ax[1], t[0:-1], np.mean(ys_d, axis=0), np.std(ys_d, axis=0))
# ax[1].plot(t[:-1], y_d)
plt.show()