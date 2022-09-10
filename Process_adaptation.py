import os
import csv
import numpy as np
from matplotlib import pyplot as plt
import pickle
import utils
import re
import obs_utils
from itertools import zip_longest

def plot_traj(traj):
    fig = plt.figure(constrained_layout=True)
    rpy_subfig = [["r"], ["p"], ["y"]]
    act_subfig = [["a"]]
    ax = fig.subplot_mosaic([[rpy_subfig, act_subfig]])
    unrolled = np.array(list(map(lambda obs_act: obs_utils.unroll_obs(obs_act[0]), traj))).T
    if len(traj) > 50:
        try:
            # obs_utils.convert_traj_to_flight_log(traj, paths.traj_path())
            er = unrolled[0]
            ep = unrolled[1]
            ey = unrolled[2]
            r = unrolled[3]
            p = unrolled[4]
            y = unrolled[5]
            sr = er + r
            sp = ep + p
            sy = ey + y

            ax["r"].plot(r)
            ax["r"].set_ylim(-200,200)
            ax["p"].plot(p)
            ax["p"].set_ylim(-200,200)
            ax["y"].plot(y)
            ax["y"].set_ylim(-200,200)

            ax["r"].plot(sr)
            ax["p"].plot(sp)
            ax["y"].plot(sy)

            ax["a"].plot(unrolled[-4])
            ax["a"].plot(unrolled[-3])
            ax["a"].plot(unrolled[-2])
            ax["a"].plot(unrolled[-1])
            ax["a"].set_ylim(-1,1)
            return fig
        except:
            print("err")
    else:
        print("trajectory too short: ", len(traj))

def string_to_digits(s):
    return ''.join(filter(str.isdigit, s))

def dirs_and_nums(path):
    dirs = list(filter(lambda d: os.path.isdir(os.path.join(path, d)), os.listdir(path)))
    dir_nums = map(lambda dir_digits: (dir_digits[0], int(dir_digits[1])), filter(lambda dir_digits: dir_digits[1] != "", zip(dirs, map(string_to_digits, dirs))))
    sorted_ckpts_and_nums = sorted(dir_nums, key=lambda x: x[1])
    return map(lambda ckpt_num: (os.path.join(path, ckpt_num[0]), ckpt_num), sorted_ckpts_and_nums)

def trajes_from_ckpt(path):
    trajes = []
    for traj_folder, traj_num in dirs_and_nums(path):
        for f in os.listdir(traj_folder):
            if f.split(".")[1] == "p":
                trajes.append(os.path.join(traj_folder, f))
    return trajes

def get_traj_paths_from_ckpts(path):
    trajes = []
    for ckpt_folder, ckpt_num in dirs_and_nums(path):
        trajes += trajes_from_ckpt(ckpt_folder)
    return trajes

def trajes_data(paths):
    return list(filter(lambda traj: len(traj) > 20, map(lambda trajpath: pickle.load(open(trajpath, "rb")), paths)))

def obs_to_errors(obs_action):
    obs, action = obs_action
    return np.abs(obs.error.yaw), np.mean(np.abs([obs.error.pitch, obs.error.roll]))
# def trajes_to_infos

def traj_smoothnesses(traj):
    acts = np.array(list(map(lambda obs_act: obs_utils.unroll_act(obs_act[1]), traj)))*0.5+0.5
    return utils.motors_smoothness(acts)

def traj_mae(traj):
    return np.mean(np.array(list(map(obs_to_errors, traj))), axis=0)

def progress_plot_adapt_ckpts(path):
    data = trajes_data(get_traj_paths_from_ckpts(path))
    plt.plot(list(map(traj_smoothnesses, data)))
    plt.show()    
    plt.plot(list(map(traj_mae, data)))
    plt.show()

def progress_plot_adapt_trajs(path):
    data = trajes_data(trajes_from_ckpt(path))
    plt.plot(list(map(traj_smoothnesses, data)))
    plt.show()    
    plt.plot(list(map(traj_mae, data)))
    plt.show()

def plot_progress_from_trajess(trajess1, trajess2):
    f, ax = plt.subplots(1,2, sharex=True, constrained_layout=True, figsize=(6, 1.6))
    def plot_transposed(ax, traj_fn, trajess):
        transposed = list(zip_longest(*list(map(lambda trajes: list(map(traj_fn, trajes)), trajess))))
        means = np.array(list(map(lambda curr_step_data: np.mean(list(filter(lambda x: x is not None, curr_step_data))), transposed)))
        stds = np.array(list(map(lambda curr_step_data: np.std(list(filter(lambda x: x is not None, curr_step_data))), transposed)))
        means = means[:30]
        stds = stds[:30]
        return utils.plot_with_std(ax, list(range(len(means))), means, stds)
    legend_anchor = plot_transposed(ax[0], traj_mae, trajess1)
    legend_no_anchor = plot_transposed(ax[0], traj_mae, trajess2)
    plot_transposed(ax[1], traj_smoothnesses, trajess1)
    plot_transposed(ax[1], traj_smoothnesses, trajess2)
    # ax[0,0].set_ylim(-25,100)
    # ax[0,1].set_ylim(-25,100)
    # ax[0,1].set_yticklabels([])
    # ax[1,1].set_yticklabels([])
    ax[0].set_ylim(0,100)
    ax[1].set_ylim(0,0.01)
    ax[0].set_ylabel("MAE (deg/s) $\\pm \\sigma$")
    ax[1].set_ylabel("Smoothness $\\pm \\sigma$")
    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position("right")
    ax[1].legend([legend_anchor,legend_no_anchor], ["With Anchor", "Without Anchor"], loc="upper right")
    
    # ax[1,0].set_xlabel("Adaptation step")
    # ax[1,1].set_xlabel("Adaptation step")

    # ax[0,0].title.set_text('(a) $\\Psi$DDPG$\\times$')
    # ax[0,1].title.set_text('(b) DDPG$\\times$')
    # ax[0,1].tick
    plt.savefig("plots/reality/with_vs_without_ac.pdf")
    plt.show()
    # for trajes in trajess:
    #     plt.plot(list(map(traj_mae, trajes)))
    # plt.show()


def get_trajess(path):
    return list(map(lambda folder: trajes_data(get_traj_paths_from_ckpts(os.path.join(path, folder))), os.listdir(path)))
        

def plot_withAC():
    plot_progress_from_trajess(get_trajess("data/reality/Adaptations/Ckpts/WithAC"))

def plot_withoutAC():
    plot_progress_from_trajess(get_trajess("data/reality/Adaptations/Ckpts/WithoutAC"))

def plot_AC_vs_no_AC():
    plot_progress_from_trajess(
        get_trajess("data/reality/Adaptations/Ckpts/WithAC"),
        get_trajess("data/reality/Adaptations/Ckpts/WithoutAC")
    )    

if __name__ == '__main__':
    # progress_plot_adapt_trajs("data/reality/Adaptations/Trajes/OnStringAdaptation2")
    # progress_plot_adapt_trajs("data/reality/Adaptations/Trajes/OnStringAdaptation4")
    # progress_plot_adapt_trajs("data/reality/Adaptations/Trajes/OnStringAdaptation5")
    # progress_plot_adapt_trajs("data/reality/Adaptations/Trajes/OnStringAdaptation6")
    # progress_plot_adapt_ckpts("data/reality/Adaptations/Ckpts/OnStringAdaptation")
    # progress_plot_adapt_ckpts("data/reality/Adaptations/Ckpts/OnStringAdaptation3")
    # plot_withAC()
    # plot_withoutAC()
    plot_AC_vs_no_AC()
    
    # for folder in os.listdir("data/reality/Adaptations/Ckpts/WithAC"):
    #     print(folder)
    #     progress_plot_adapt_ckpts("data/reality/Adaptations/Ckpts/WithAC/"+folder)
    # plot_traj(pickle.load(open("data/reality/OnStringAdaptation/ckpt_0.0/traj_0.0/traj.p", "rb")))
    # plot_traj(pickle.load(open("data/reality/OnStringAdaptation/ckpt_8.0/traj_2.0/traj.p", "rb")))
    # plt.show()