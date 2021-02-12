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

figsize=(8,5)
xlim = [30,60]

def plot_following(ax, index):
    ax.plot(t, set_points[:,index],"-", label="desired", linewidth=3, alpha=1, color="black")
    ax.plot(t, real_gyros[:,index], label="reality", linewidth=1, alpha=1, linestyle="-", color=utils.colors[5])
    ax.plot(t, sim_gyros[:,index], linewidth=1, alpha=1, label="simulation", linestyle="-.", color=utils.colors[1])


def left_plot():
    fig, ax = plt.subplots(3, sharex=True, figsize=figsize)

    plot_following(ax[0], 0)
    plot_following(ax[1], 1)
    plot_following(ax[2], 2)

    ax[0].legend(ncol = 3, loc='upper center', bbox_to_anchor=(0.5, 1.38), columnspacing=0.8)
    ax[0].set_ylabel("Roll (deg/s)")
    ax[1].set_ylabel("Pitch (deg/s)")
    ax[2].set_ylabel("Yaw (deg/s)")

    ax[2].set_xlabel("Time (s)")
    ax[2].set_xlim(xlim)

    fig.align_ylabels()

    plt.subplots_adjust(left=0.1, right=0.98, top=0.9)
    # plt.show()
    plt.savefig("plots/gap/following_left.pdf")



def right_plot():
    fig, ax = plt.subplots(1, sharex=True, figsize=figsize)

    ax.plot(t, errors[:,0], color=utils.colors[3], label="roll", linestyle="-.", linewidth=0.8, alpha=0.7)
    ax.plot(t, errors[:,1], color=utils.colors[2], label="pitch", linestyle="-.", linewidth=0.8, alpha=0.7)
    ax.plot(t, errors[:,2], color=utils.colors[0], label="yaw", linestyle="-.", linewidth=0.8, alpha=0.7)
    # ax.legend()

    ax.legend(ncol = 3, loc='upper center', bbox_to_anchor=(0.5, 1.11), columnspacing=0.8)

    ax.set_xlabel("Time (s)")
    ax.set_xlim(xlim)
    ax.set_ylim([-1, 150])
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_ylabel("SPG (deg/s)")

    fig.align_ylabels()

    # plt.savefig("plots/gap/following.pdf")
    plt.subplots_adjust(left=0.02, right=0.9, top=0.9)
    # plt.show()
    plt.savefig("plots/gap/following_right.pdf")

left_plot()
right_plot()