import pickle
import utils
from matplotlib import pyplot as plt
import numpy as np

from_index, results = pickle.load(open("data/reality_gap/baselines_f81876a_200416-004611_rs157873_following.pkl" , "rb"))
errors, sim_gyros, real_gyros, set_points = results
t = from_index["t"]
errors = np.abs(sim_gyros - real_gyros)[:len(t)]
print(errors.shape)
print(np.mean(errors, axis=0), np.std(errors, axis=0))
sim_gyros = sim_gyros[:len(t)]
real_gyros = real_gyros[:len(t)]
set_points = set_points[:len(t)]

fig, ax = plt.subplots(4, sharex=True)

print(sim_gyros.shape)
print(t.shape)
def plot_following(index):
    ax[index].plot(t, set_points[:,index],"-", label="desired", linewidth=3, alpha=1, color="black")
    ax[index].plot(t, real_gyros[:,index], label="reality", linewidth=1, alpha=1, linestyle="-", color=utils.colors[5])
    ax[index].plot(t, sim_gyros[:,index], linewidth=1, alpha=1, label="simulation", linestyle="-.", color=utils.colors[1])
plot_following(0)
plot_following(1)
plot_following(2)

ax[3].plot(t, errors[:,0], color=utils.colors[3], label="roll", linestyle="-.", linewidth=0.8)
ax[3].plot(t, errors[:,1], color=utils.colors[2], label="pitch", linestyle="-.", linewidth=0.8)
ax[3].plot(t, errors[:,2], color=utils.colors[0], label="yaw", linestyle="-.", linewidth=0.8)
ax[3].legend()

ax[0].legend(ncol = 3, loc='upper center', bbox_to_anchor=(0.5, 1.5), columnspacing=0.8)
ax[0].set_ylabel("Roll")
ax[1].set_ylabel("Pitch")
ax[2].set_ylabel("Yaw")
ax[3].set_ylabel("Reality Gap")

ax[3].set_xlabel("Time (s)")

fig.align_ylabels()

# plt.savefig("plots/gap/following.pdf")
plt.show()

