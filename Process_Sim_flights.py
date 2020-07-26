import numpy as np
from matplotlib import pyplot as plt

def plot_motor(desired_rpy, rpy, motor_ouptuts):
    f, ax = plt.subplots(1,1, sharex=True, sharey=False)

    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    t = np.array(range(len(desired_rpy[:,0]))) * 0.001
    colors = ['#e41a1c', '#3673a7', '#4daf4a', '#984ea3', '#6e0178', '#ff7f00']

    def plot_desired_vs_actual(i,label):
        desired = ax[i].plot(t, desired_rpy[:,i],"r--", label="desired", linewidth=2, alpha=1, color=colors[0])
        no_reg = ax[i].plot(t, rpy[0,:,i], linewidth=1.5, alpha=1, label="PPO",linestyle="-.", color=colors[1])
        reg = ax[i].plot(t, rpy[1,:,i], linewidth=1, alpha=1, label="PPO + SRC", color="black")
        # rpy_mean = np.mean(rpy[2:,:,i],axis=0)
        # ours=ax[i].plot(t, rpy_mean, linewidth=1.5, alpha=1, linestyle="-", color=colors[2])
        # ax[i].fill_between(t, rpy_mean-rpy_std[:,i], rpy_mean+rpy_std[:,i], color=colors[2],alpha=0.5)
        # between = ax[i].fill(np.NaN, np.NaN, alpha=0.5, color = colors[2])
        ax[i].set_ylabel(label)
        if i == 0:
            ax[0].legend([desired[0],no_reg[0], reg[0]], ["Desired", "PPO", "PPO + SRC"], loc="right")
        # ax[i].legend([desired[0], (p2[0], line[0])], ["Desired", "Actual (and $\\sigma$)"], loc="right")
    # plot_desired_vs_actual(0, "Roll (deg/s)")
    # plot_desired_vs_actual(1, "Pitch (deg/s)")
    # plot_desired_vs_actual(2, "Yaw (deg/s)")

    ax.set_ylabel("PPO (Motor Usage %)")
    # ax[1].set_ylabel("PPO + SRC (Motor Usage %)")
    ax.yaxis.set_label_position("left")
    ax.yaxis.tick_right()
    # ax[1].yaxis.set_label_position("left")
    # ax[1].yaxis.tick_right()
    colors = [colors[1], colors[4]]
    for i in range(2):
        print(i)
        m0 = motor_ouptuts[i,:,0]*100
        m1 = motor_ouptuts[i,:,1]*100
        m2 = motor_ouptuts[i,:,2]*100
        m3 = motor_ouptuts[i,:,3]*100
        m = [m0]
        
        lines = ["-", "-.", ":", "--"]
        for j in range(len(m)):
            ax.plot(t, m[j], label="Motor {}".format(j+1), linestyle=lines[j], alpha=1, color=colors[i])
    # ax[1].set_ylim(0, 100)
    ax.legend(loc="right")
    f.align_ylabels()
    # ax[1].set_xlabel("Time (s)")

    return f, ax


def plot_following(desired_rpy, rpy, motor_ouptuts):
    rpy_std = np.std(rpy[2:,:,:],axis=0)

    f, ax = plt.subplots(3,2, sharex=True, sharey=False)

    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    t = np.array(range(len(desired_rpy[:,0]))) * 0.001
    colors = ['#e41a1c', '#265285', '#4daf4a', '#984ea3', '#6e0178', '#ff7f00']

    def plot_desired_vs_actual(i,label):
        desired = ax[i][0].plot(t, desired_rpy[:,i],"r--", label="desired", linewidth=2, alpha=1, color=colors[0])
        pid = ax[i][0].plot(t, rpy[0,:,i], linewidth=1.5, alpha=1, label="pid",linestyle=":", color=colors[1])
        neuroflight = ax[i][0].plot(t, rpy[1,:,i], linewidth=1, alpha=1, label="Neuroflight", linestyle="-.", color=colors[3])
        rpy_mean = np.mean(rpy[2:,:,i],axis=0)
        ours=ax[i][0].plot(t, rpy_mean, linewidth=1.5, alpha=1, linestyle="-", color="black")
        ax[i][0].fill_between(t, rpy_mean-rpy_std[:,i], rpy_mean+rpy_std[:,i], color=colors[2],alpha=0.5)
        between = ax[i][0].fill(np.NaN, np.NaN, alpha=0.5, color = colors[2])
        ax[i][0].set_ylabel(label)
        if i == 0:
            ax[0][0].legend([desired[0],pid[0], neuroflight[0], (between[0], ours[0])], ["Desired", "PID", "Neuroflight", "PPO+CAPS"], loc="right")

    plot_desired_vs_actual(0, "Roll (deg/s)")
    plot_desired_vs_actual(1, "Pitch (deg/s)")
    plot_desired_vs_actual(2, "Yaw (deg/s)")

    def plot_motor_outputs(i, label):
        ax[i][1].set_ylabel(label)
        ax[i][1].yaxis.set_label_position("right")
        ax[i][1].yaxis.tick_right()
        lines = ["-", "-.", ":", "--"]
        for j in range(4):
            ax[i][1].plot(t, motor_ouptuts[i, :, j], label="Motor {}".format(j+1), linestyle=lines[j], alpha=0.8)

    plot_motor_outputs(0, "PID (%)")
    plot_motor_outputs(1, "Neuroflight (%)")
    plot_motor_outputs(2, "PPO+CAPS (%)")

    ax[0][1].legend(loc="right")
    f.align_ylabels()
    ax[2][1].set_xlabel("Time (s)")

    return f, ax


if __name__ == "__main__":
    import pickle
    (desired_rpy, rpy, motor_outputs) = pickle.load(open("data/simulation/wil_pid_caps.p", "rb"))
    print(rpy.shape)
    plot_following(desired_rpy, rpy, motor_outputs)
    plt.show()
