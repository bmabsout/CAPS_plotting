import pickle
import utils
from matplotlib import pyplot as plt
import numpy as np

windows_with_errors = pickle.load(open("data/reality_gap/baselines_f81876a_200416-004611_rs157873_gap.pkl" , "rb"))


ys = utils.to_array_truncate(list(map(lambda w: np.array(w["errors"]), windows_with_errors)))
print(ys.shape)
# ys = ys[:, :1000]
y = np.mean(ys, axis=0)
y_std = np.std(ys, axis=0)
x = windows_with_errors[0]["t"][:len(y)]
fig, ax = plt.subplots()
utils.plot_with_std(ax, x, y, y_std)
plt.show()