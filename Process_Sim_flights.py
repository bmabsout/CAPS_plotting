import numpy as np
from matplotlib import pyplot as plt
import utils

t_diff = 0.00137

def plot_motor(desired_rpy, rpy, motor_ouptuts):
    f, ax = plt.subplots(2,1, sharex=True, sharey=False)

    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    t = np.array(range(len(desired_rpy[:,0]))) * t_diff

    # ax[1].set_ylabel("PPO + SRC (Motor Usage %)")
    # ax[1].yaxis.set_label_position("left")
    # ax[1].yaxis.tick_right()
    for i in range(motor_outputs.shape[0]):
        for j in range(motor_ouptuts.shape[2]):
            ax[i].plot(t, motor_outputs[i,:, j]*100, label="Motor {}".format(j+1), linestyle=utils.line_styles[j], alpha=1., color=utils.colors[j])
        ax[i].set_ylabel("PPO (Motor Usage %)")
        ax[i].yaxis.set_label_position("left")
        ax[i].yaxis.tick_right()
        ax[i].legend(loc="right")
    # ax[1].set_ylim(0, 100)
    f.align_ylabels()
    # ax[1].set_xlabel("Time (s)")

    return f, ax


def plot_following(desired_rpy, rpy, motor_ouptuts):
    rpy_std = np.std(rpy[2:,:,:],axis=0)

    f, ax = plt.subplots(3,2, sharex=True, sharey=False)

    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    t = np.array(range(len(desired_rpy[:,0]))) * t_diff
    # colors = ['#e41a1c', '#265285', '#4daf4a', '#984ea3', '#6e0178', '#ff7f00']

    def plot_desired_vs_actual(i,label):
        desired = ax[i][0].plot(t, desired_rpy[:,i],"r--", label="desired", linewidth=2, alpha=1, color=utils.colors[0])
        pid = ax[i][0].plot(t, rpy[0,:,i], linewidth=1.5, alpha=1, label="pid",linestyle=":", color=utils.colors[1])
        neuroflight = ax[i][0].plot(t, rpy[1,:,i], linewidth=1, alpha=1, label="Neuroflight", linestyle="-.", color=utils.colors[3])
        rpy_mean = np.mean(rpy[2:,:,i],axis=0)
        ours=ax[i][0].plot(t, rpy_mean, linewidth=1.5, alpha=1, linestyle="-", color="black")
        ax[i][0].fill_between(t, rpy_mean-rpy_std[:,i], rpy_mean+rpy_std[:,i], color=utils.colors[2],alpha=0.5)
        between = ax[i][0].fill(np.NaN, np.NaN, alpha=0.5, color = utils.colors[2])
        ax[i][0].set_ylabel(label)
        if i == 0:
            ax[0][0].legend([desired[0],pid[0], neuroflight[0], (between[0], ours[0])], ["Desired", "PID", "Neuroflight", "PPO+CAPS"], loc="right")
        mae = np.mean(np.abs(rpy[2:,:,i] - desired_rpy[:,i]), axis=1)
        print("MAE:",label, np.mean(mae))
        print("MAE std:",label, np.std(mae))
        return mae

    mae1 = plot_desired_vs_actual(0, "Roll (deg/s)")
    mae2 = plot_desired_vs_actual(1, "Pitch (deg/s)")
    mae3 = plot_desired_vs_actual(2, "Yaw (deg/s)")
    print(mae1.shape)
    print("all:", np.std(np.vstack([mae1, mae2, mae3])))

    def plot_motor_outputs(i, label):
        ax[i][1].set_ylabel(label)
        ax[i][1].yaxis.set_label_position("right")
        ax[i][1].yaxis.tick_right()
        lines = ["-", "-.", ":", "--"]
        for j in range(4):
            ax[i][1].plot(t, motor_outputs[i, :, j], label="Motor {}".format(j+1), linestyle=lines[j], alpha=0.8)

    plot_motor_outputs(0, "PID (%)")
    plot_motor_outputs(1, "Neuroflight (%)")
    plot_motor_outputs(2, "PPO+CAPS (%)")
    ax[0][1].legend(loc="right")
    f.align_ylabels()
    ax[2][1].set_xlabel("Time (s)")

    return f, ax


def plot_rpy(t, rpy_algs_dict, parent_grid):
    grid = parent_grid.subgridspec(len(rpy_algs_dict), 1, hspace=0,wspace=0)
    ax = None
    for i, (rpy_label, algs_dict) in enumerate(rpy_algs_dict.items()):
        ax = plt.subplot(grid[i], sharex=ax)
        ax.set_ylabel(rpy_label)
        for j, (alg_label, (alg, kwargs)) in enumerate(algs_dict.items()):
            ax.plot(t, alg, label=alg_label, **kwargs)
    return ax

def plot_motor_outputs(t, motor_outputs, parent_grid, labels, sharex):
    grid = parent_grid.subgridspec(len(labels), 1, hspace=0,wspace=0)
    ax = None
    for i, label in enumerate(labels):
        ax = plt.subplot(grid[i],sharex=sharex, sharey=ax)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        for motor in range(motor_outputs.shape[2]):
            ax.plot(t, motor_outputs[i, :, motor], label=f"Motor {motor+1}", color=utils.colors[motor], linestyle=utils.line_styles[motor], linewidth=0.8)
        ax.set_ylabel(label)
    return ax



def plot_reg_vs_no_reg(desired_rpy, rpy, motor_outputs, labels):
    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(1, 2)
    t = np.array(range(len(desired_rpy[:,0]))) * t_diff
    rpy_algs_dict = {}
    rpy_labels = ["Roll (deg/s)", "Pitch (deg/s)", "Yaw (deg/s)"]
    for i, rpy_label in enumerate(rpy_labels):
        algs_dict = {}
        algs_dict["Desired"] = (desired_rpy[:, i], {"color": "black", "linewidth":1})
        for j, label in enumerate(labels):
            algs_dict[label] = (rpy[j, :, i], {"color": utils.colors[j], "linestyle": utils.line_styles[j+1], "linewidth": 0.8})
        rpy_algs_dict[rpy_label] = algs_dict

    ax_rpy = plot_rpy(t, rpy_algs_dict, gs[0])
    ax_rpy.set_xlabel("Time (s)")
    ax_rpy.legend(loc='lower center', bbox_to_anchor=(.5, -0.85),
          ncol=4, fancybox=True, shadow=True)

    motor_labels = list(map(lambda label: label + " (%)", labels))
    ax_motor = plot_motor_outputs(t, motor_outputs*100, gs[1], motor_labels, ax_rpy)
    ax_motor.set_xlabel("Time (s)")
    ax_motor.legend(loc='upper center', bbox_to_anchor=(.5, -0.29),
          ncol=4, fancybox=True, shadow=True, columnspacing=0.8)
    fig.align_ylabels()

def motor_smoothness(motor_outputs):
    print(motor_outputs.shape)
    pid = motor_outputs[0]
    wil = motor_outputs[1]
    caps = motor_outputs[2:]
    def smoothness(motors):
        smoothnesses = []
        for i in range(motors.shape[1]):
            freqs, amplitudes = utils.fourier_transform(motors[:,i]*2-1,1)
            smoothnesses.append(utils.smoothness(amplitudes))
        return np.mean(smoothnesses)
    print("wil:",smoothness(wil))
    print("pid:",smoothness(pid))
    caps_smoothnesses = []
    for i in range(caps.shape[0]):
        caps_smoothnesses.append(smoothness(caps[i]))
    print("caps:", np.mean(caps_smoothnesses), np.std(caps_smoothnesses))


if __name__ == "__main__":
    import pickle
    # (desired_rpy, rpy, motor_outputs) = pickle.load(open("data/simulation/pid_wil_caps.p", "rb"))
    # plot_following(desired_rpy, rpy, motor_outputs)

    # (desired_rpy, rpy, motor_outputs) = pickle.load(open("data/simulation/ppo+caps.p", "rb"))
    # plot_motor(desired_rpy,rpy, motor_outputs)

    (desired_rpy, rpy, motor_outputs) = pickle.load(open("data/simulation/reg_vs_no_reg.p", "rb"))
    plot_reg_vs_no_reg(desired_rpy, rpy, motor_outputs, ["PPO","PPO+CAPS"])

    # print(rpy.shape)
    plt.show()

    # (desired_rpy, rpy, motor_outputs) = pickle.load(open("data/simulation/pid_wil_caps.p", "rb"))
    # motor_smoothness(motor_outputs)

