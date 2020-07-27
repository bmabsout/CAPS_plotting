import pickle
import matplotlib.pyplot as plt
import numpy as np
import utils
import os


def plot_following(files, ax_top, ax_bottom):
    # colors = ['#e41a1c', '#265285', '#4daf4a', '#984ea3', '#6e0178', '#ff7f00']
    colors = utils.colors
    handles = []
    for i, file in enumerate(files):
        (feedback_list, setpoint_list, time_list, outputs, alg_name) = pickle.load(open(file, "rb"))
        if i == 0:
            ax_top.plot(time_list, setpoint_list, color='black', label='Target')
        line = ax_top.plot(time_list, feedback_list, color=colors[i-1], alpha=0.8, label=alg_name)
        ax_top.grid(True)

        ax_bottom.plot(time_list, outputs, color=colors[i-1], alpha=0.8, label=alg_name)
        ax_bottom.set_xlabel('Time')
        ax_bottom.grid(True)
        handles.append(line[0])
    return handles

def plot_followings(folders, parent_grid, labels):
    axs = []
    ax_top = None
    ax_bottom = None
    grid = parent_grid.subgridspec(2, len(folders), hspace=0,wspace=0)
    for i, folder in enumerate(folders):
        ax_top = plt.subplot(grid[0,i], sharey=ax_top, sharex=ax_bottom)
        ax_bottom = plt.subplot(grid[1,i], sharey=ax_bottom, sharex=ax_top)
        plt.setp(ax_top.get_xticklabels(), visible=False)
        axs.append({'top': ax_top, 'bottom': ax_bottom})
        if i%2 == 0:
            ax_top.set_title(f"{labels[i]}) Without CAPS")
        else:
            ax_top.set_title(f"{labels[i]}) With CAPS")
            plt.setp(ax_top.get_yticklabels(), visible=False)
            plt.setp(ax_bottom.get_yticklabels(), visible=False)
        files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.p') and not (file == "TRPO.p" or file == "VPG.p")]
        handles = plot_following(files, ax_top, ax_bottom)
    return axs, handles


if __name__ == "__main__":
    import sys
    fig = plt.figure(figsize=(6, 6))
    labels = ["a", "b", "c", "d"]
    folders = sys.argv[1:]
    gs = fig.add_gridspec(1, 2, hspace=0.1, wspace=0.07, left=0.07, bottom=0.24, top=0.92, right=0.99)

    axs, handles = plot_followings(folders[0:2], gs[0], labels[0:2])
    plot_followings(folders[2:4], gs[1], labels[2:4])
    axs[0]['top'].set_ylabel('State')
    axs[0]['bottom'].set_ylabel('Action')
    fig.legend(handles = handles, loc='lower center', bbox_to_anchor=(.5, .0),
          ncol=5, fancybox=True, shadow=True)
    fig.align_ylabels()

    plt.show()
