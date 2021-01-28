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

    # plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    t = np.array(range(len(desired_rpy[:,0]))) * t_diff
    # colors = ['#e41a1c', '#265285', '#4daf4a', '#984ea3', '#6e0178', '#ff7f00']

    def plot_desired_vs_actual(i,label):
        desired = ax[i][0].plot(t, desired_rpy[:,i],"-", label="desired", linewidth=3, alpha=1, color="black")
        pid = ax[i][0].plot(t, rpy[0,:,i], linewidth=1, alpha=1, label="pid",linestyle="-", color=utils.colors[0])
        neuroflight = ax[i][0].plot(t, rpy[1,:,i], linewidth=1, alpha=1, label="Neuroflight", linestyle="-.", color=utils.colors[1])
        rpy_mean = np.mean(rpy[2:,:,i],axis=0)
        ours=ax[i][0].plot(t, rpy_mean, linewidth=1.5, alpha=1, linestyle="-", color=utils.colors[2])
        ax[i][0].fill_between(t, rpy_mean-rpy_std[:,i], rpy_mean+rpy_std[:,i], color=utils.colors[2],alpha=0.5)
        between = ax[i][0].fill(np.NaN, np.NaN, alpha=0.5, color = utils.colors[2])
        ax[i][0].set_ylabel(label)
        if i == 0:
            ax[0][0].legend([desired[0],pid[0], neuroflight[0], (between[0], ours[0])], ["Desired", "PID", "Neuroflight", "PPO+CAPS"], loc="right")
        mae = np.mean(np.abs(rpy[2:,:,i] - desired_rpy[:,i]), axis=1)
        print(mae)
        print("MAE:",label, np.mean(mae))
        print("MAE std:",label, np.std(mae))
        return mae

    mae1 = plot_desired_vs_actual(0, "Roll (deg/s)")
    mae2 = plot_desired_vs_actual(1, "Pitch (deg/s)")
    mae3 = plot_desired_vs_actual(2, "Yaw (deg/s)")
    print(mae1.shape)
    print("all:", np.mean(np.vstack([mae1, mae2, mae3])))
    print("all std:", np.std(np.vstack([mae1, mae2, mae3])))

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
    ax[2][0].set_xlabel("Time (s)")

    return f, ax


def plot_rpy(t, rpy_algs_dict, parent_grid):
    grid = parent_grid.subgridspec(len(rpy_algs_dict), 1, hspace=0.1,wspace=0)
    ax = None
    axs = []
    for i, (rpy_label, algs_dict) in enumerate(rpy_algs_dict.items()):
        ax = plt.subplot(grid[i])
        ax.set_ylabel(rpy_label)
        for j, (alg_label, (alg, kwargs)) in enumerate(algs_dict.items()):
            ax.plot(t, alg, label=alg_label, **kwargs)
        axs.append(ax)

    axs[0].get_shared_y_axes().join(*axs)
    axs[0].get_shared_x_axes().join(*axs)

    for ax in axs[:-1]:
        ax.set_xticklabels([])
    return axs

def smoothness(motors):
    smoothnesses = []
    for i in range(motors.shape[1]):
        freqs, amplitudes = utils.fourier_transform(motors[:,i]*2-1,1)
        smoothnesses.append(utils.smoothness(amplitudes))
    return np.mean(smoothnesses)


def plot_motor_outputs(t, motor_outputs, parent_grid, labels, sharex, label_pos="right"):
    grid = parent_grid.subgridspec(len(labels), 1, hspace=0.1,wspace=0)
    ax = None
    axs = []
    for i, label in enumerate(labels):
        ax = plt.subplot(grid[i], sharey = ax)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position(label_pos)
        for motor in range(motor_outputs.shape[2]):
            ax.plot(t, motor_outputs[i, :, motor], label=f"Motor {motor+1}", color=utils.colors[motor], linestyle=utils.line_styles[motor], linewidth=0.8)
        ax.set_ylabel(label)
        axs.append(ax)
    axs[0].get_shared_x_axes().join(*(axs + ([sharex] if sharex else [])))
    for ax in axs[:-1]:
        ax.set_xticklabels([])
    # for i in motor_outputs.shape[]
    print("Smoothness_no_reg:", smoothness(motor_outputs[0]))
    print("Smoothness_reg:", smoothness(motor_outputs[1]))
    return axs


def plot_reg_vs_no_reg(desired_rpy, rpy, motor_outputs, labels):
    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(1, 2, left=0.07, bottom=0.21, right=0.94, top = 0.943, wspace=0.05)
    t = np.array(range(len(desired_rpy[:,0]))) * t_diff
    rpy_algs_dict = {}
    rpy_labels = ["Roll (deg/s)", "Pitch (deg/s)", "Yaw (deg/s)"]
    for i, rpy_label in enumerate(rpy_labels):
        algs_dict = {}
        algs_dict["Desired"] = (desired_rpy[:, i], {"color": "black", "linewidth":1.5, "linestyle": ':'})
        for j, (label, color, linestyle) in enumerate(labels):
            algs_dict[label] = (rpy[j, :, i], {"color": color, "linestyle": linestyle, "linewidth": 1})
        rpy_algs_dict[rpy_label] = algs_dict

    axs_rpy = plot_rpy(t, rpy_algs_dict, gs[0])

    motor_labels = list(map(lambda label: label[0] + " (%)", labels))
    axs_motor = plot_motor_outputs(t, motor_outputs*100, gs[1], motor_labels, axs_rpy[-1])

    axs_rpy[-1].set_xlabel("Time (s)")
    axs_motor[-1].set_xlabel("Time (s)")

    axs_rpy[-1].legend(loc='lower left', bbox_to_anchor=(0.1, -0.9),
          ncol=4, fancybox=True, shadow=True, columnspacing=0.8)
    axs_motor[-1].legend(loc='lower left', bbox_to_anchor=(0, -0.59),
          ncol=4, fancybox=True, shadow=True, columnspacing=0.8)

    axs_rpy[0].set_title("a) RPY tracking")
    axs_motor[0].set_title("b) Simulated motor usage")

    fig.align_ylabels()


def plot_reg_only(desired_rpy, rpy, motor_outputs, labels):
    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(1, 1)#, left=0.07, bottom=0.21, right=0.94, top = 0.943, wspace=0.05)
    t = np.array(range(len(desired_rpy[:,0]))) * t_diff
    rpy_algs_dict = {}
    rpy_labels = ["Roll (deg/s)", "Pitch (deg/s)", "Yaw (deg/s)"]
    for i, rpy_label in enumerate(rpy_labels):
        algs_dict = {}
        algs_dict["Desired"] = (desired_rpy[:, i], {"color": "black", "linewidth":1.5, "linestyle": ':'})
        for j, (label, color, linestyle) in enumerate(labels):
            algs_dict[label] = (rpy[j, :, i], {"color": color, "linestyle": linestyle, "linewidth": 1})
        rpy_algs_dict[rpy_label] = algs_dict

    axs_rpy = plot_rpy(t, rpy_algs_dict, gs[0])
    for ax in axs_rpy:
        ax.set_xlabel(None)
        ax.set_xticks([])
        ax.yaxis.tick_right()

    # motor_labels = list(map(lambda label: label[0] + " (%)", labels))
    # axs_motor = plot_motor_outputs(t, motor_outputs*100, gs[0], motor_labels)

    axs_rpy[-1].set_xlabel("Time (s)")
    # axs_motor[-1].set_xlabel("Time (s)")

    axs_rpy[0].legend(loc='upper center', 
          ncol=4, fancybox=True, shadow=False, columnspacing=0.8, framealpha=0.8)
    # axs_motor[-1].legend(loc='lower center', bbox_to_anchor=(0.5, -1.0),
          # ncol=4, fancybox=True, shadow=True, columnspacing=0.8)

    axs_rpy[0].set_title("a) RPY tracking")
    # axs_motor[0].set_title("b) Simulated motor usage")

    fig.align_ylabels()

def plot_no_reg_only(desired_rpy, rpy, motor_outputs, labels):
    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(1, 1)#, left=0.07, bottom=0.21, right=0.94, top = 0.943, wspace=0.05)
    t = np.array(range(len(desired_rpy[:,0]))) * t_diff
    rpy_algs_dict = {}
    rpy_labels = ["Roll (deg/s)", "Pitch (deg/s)", "Yaw (deg/s)"]
    for i, rpy_label in enumerate(rpy_labels):
        algs_dict = {}
        algs_dict["Desired"] = (desired_rpy[:, i], {"color": "black", "linewidth":1.5, "linestyle": ':'})
        for j, (label, color, linestyle) in enumerate(labels):
            algs_dict[label] = (rpy[j, :, i], {"color": color, "linestyle": linestyle, "linewidth": 1})
        rpy_algs_dict[rpy_label] = algs_dict

    # axs_rpy = plot_rpy(t, rpy_algs_dict, gs[0])

    motor_labels = list(map(lambda label: label[0] + " (%)", labels))
    axs_motor = plot_motor_outputs(t, motor_outputs*100, gs[0], motor_labels, None, "left")

    # axs_rpy[-1].set_xlabel("Time (s)")
    axs_motor[-1].set_xlabel("Time (s)")

    # axs_rpy[-1].legend(loc='lower center', bbox_to_anchor=(0.5, -1.0),
    #       ncol=4, fancybox=True, shadow=True, columnspacing=0.8)
    axs_motor[1].legend(loc='upper center',
          ncol=4, fancybox=True, shadow=False, columnspacing=0.8, framealpha=0.8)

    # axs_rpy[0].set_title("a) RPY tracking")
    axs_motor[0].set_title("b) Simulated motor usage")

    fig.align_ylabels()


def motor_smoothness(motor_outputs):
    print(motor_outputs.shape)
    pid = motor_outputs[0]
    wil = motor_outputs[1]
    caps = motor_outputs[2:]
    print("wil:",smoothness(wil))
    print("pid:",smoothness(pid))
    caps_smoothnesses = []
    for i in range(caps.shape[0]):
        caps_smoothnesses.append(smoothness(caps[i]))
    print("caps:", np.mean(caps_smoothnesses), np.std(caps_smoothnesses))


def plot_progresses(all_xs, all_ys, headers):
    print(headers)
    print(all_ys.shape)
    x = np.mean(all_xs,axis=1)
    data = { ("Tracking Reward ($p_s$)", "Tracking Error $({\\bf e}_t)$"): (all_ys[:, :, 4], all_ys[:, :, 0])
            , ("Thrust Reward ($p_u$)", "Motor Output $(\\phi_t)$"): (all_ys[:, :, 5], all_ys[:, :, 1])
            , ("Smoothness Reward ($p_c$)", "Motor Acceleration $(\\Delta y_t)$"): (all_ys[:, :, 6], all_ys[:, :, 2])
            }
    # total_rws = "Total Reward", all_ys[:, 3]
    # rights
    f, ax = plt.subplots(len(data) + 1, sharex=True, sharey=False)

    # plot a single trial for all the metrics
    left_color = utils.theme["blue"]
    right_color = utils.theme["purple"]

    def plot_progress(ax, x, y, y_std, color, label):
        utils.plot_with_std(ax, x, y, y_std, color=color)
        ax.tick_params(axis='y', labelcolor=color)
        ax.set_ylabel(label, color=color)
        # ax.set_yscale('log')

    for i, ((left_label, right_label), (left_data, right_data)) in enumerate(data.items()):
        plot_progress(ax[i], x, np.mean(left_data, axis=1), np.std(left_data, axis=1), left_color, left_label)
        plot_progress(ax[i].twinx(), x, np.mean(right_data, axis=1), np.std(right_data, axis=1), right_color, right_label)
        # ax[i].set_yscale('log')

    plot_progress(ax[-1],  x, np.mean(all_ys[:,:, 3], axis=1), np.std(all_ys[:,:,3], axis=1), left_color, "Total reward")
    ax[-1].set_xlabel("TimeSteps")
    f.align_ylabels()
    f.set_size_inches(7,10)



if __name__ == "__main__":
    import pickle
    # (desired_rpy, rpy, motor_outputs) = pickle.load(open("data/simulation/ppo+caps.p", "rb"))
    # plot_following(desired_rpy, rpy, motor_outputs)

    # (desired_rpy, rpy, motor_outputs) = pickle.load(open("data/simulation/ppo+caps.p", "rb"))
    # plot_motor(desired_rpy,rpy, motor_outputs)

    (desired_rpy, rpy, motor_outputs) = pickle.load(open("data/simulation/reg_vs_no_reg.p", "rb"))
    plot_reg_vs_no_reg(desired_rpy, rpy, motor_outputs, [("PPO", utils.theme["blue"], '-.'),("PPO+CAPS", utils.theme["orange"],'-')])

    # (desired_rpy, rpy, motor_outputs) = pickle.load(open("data/simulation/reg_vs_no_reg.p", "rb"))
    # plot_reg_only(desired_rpy[4500:, :], rpy[:,4500:,:], motor_outputs[:, 4500:, :], [("Without CAPS", utils.theme["blue"], '-.'),("With CAPS", utils.theme["orange"],'-')])


    # (desired_rpy, rpy, motor_outputs) = pickle.load(open("data/simulation/reg_vs_no_reg.p", "rb"))
    # print(desired_rpy.shape, rpy.shape, motor_outputs.shape)
    # plot_no_reg_only(desired_rpy[4500:, :], rpy[:,4500:,:], motor_outputs[:, 4500:, :], [("Without CAPS", utils.theme["blue"], '-.'),("With CAPS", utils.theme["orange"],'-')])

    # print(rpy.shape)

    # (all_xs, all_ys, headers) = pickle.load(open("data/simulation/progress_agents.p","rb"))
    # plot_progresses(all_xs, all_ys, headers)
    plt.show()

    # (desired_rpy, rpy, motor_outputs) = pickle.load(open("data/simulation/pid_wil_caps.p", "rb"))
    # motor_smoothness(motor_outputs)

