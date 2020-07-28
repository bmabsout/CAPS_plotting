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
            handles.append(ax_top.plot(time_list, setpoint_list, color='black', label='Target')[0])
        line = ax_top.plot(time_list, feedback_list, color=colors[i], alpha=0.8, label=alg_name, linestyle=utils.line_styles[i], linewidth=0.8)
        ax_top.grid(True)

        ax_bottom.plot(time_list, outputs, color=colors[i], alpha=0.8, label=alg_name, linestyle=utils.line_styles[i], linewidth=0.8)
        ax_bottom.set_xlabel('Time')
        ax_bottom.grid(True)
        handles.append(line[0])
    return handles

def plot_followings(folders_and_labels, parent_grid):
    axs = []
    ax_top = None
    ax_bottom = None
    grid = parent_grid.subgridspec(2, len(folders_and_labels), hspace=0.07,wspace=0.03)
    for i, (folder, label) in enumerate(folders_and_labels):
        ax_top = plt.subplot(grid[0,i], sharey=ax_top, sharex=ax_bottom)
        ax_bottom = plt.subplot(grid[1,i], sharey=ax_bottom, sharex=ax_top)
        ax_bottom.set_yticks([0])
        ax_bottom.set_yticklabels([''])
        ax_top.set_yticks([])
        ax_bottom.set_xticks([])
        ax_top.set_xticks([])
        plt.setp(ax_top.get_xticklabels(), visible=False)
        axs.append({'top': ax_top, 'bottom': ax_bottom})
        if i%2 == 0:
            ax_top.set_title(f"{label}) Without CAPS")
        else:
            ax_top.set_title(f"{label}) With CAPS")
            plt.setp(ax_top.get_yticklabels(), visible=False)
            plt.setp(ax_bottom.get_yticklabels(), visible=False)
        files = [os.path.join(folder, file) for file in os.listdir(folder)
                  if file.endswith('.p') and not (file == "TRPO.p" or file == "VPG.p")]
        handles = plot_following(files, ax_top, ax_bottom)
    return axs, handles

def plot_perlin_vs_step():
    fig = plt.figure(figsize=(6, 6))
    labels = ["a", "b", "c", "d"]
    folders = ["perlin_low_freq_1sec", "perlin_low_freq_reg_1sec", "step", "step_smooth"]
    folders = ["data/toy/" + folder for folder in folders ]
    folders_and_labels = list(zip(folders, labels))
    gs = fig.add_gridspec(1, 2, hspace=0.1, wspace=0.07, left=0.05, bottom=0.24, top=0.9, right=0.99)

    axs, handles = plot_followings(folders_and_labels[0:2], gs[0])
    step_axs, hs = plot_followings(folders_and_labels[2:4], gs[1])
    step_axs[0]['bottom'].set_yticklabels(['0'])
    axs[0]['bottom'].get_shared_y_axes().join(axs[0]['bottom'], step_axs[0]['bottom'])
    axs[0]['bottom'].autoscale()
    axs[0]['bottom'].set_yticklabels([])
    axs[1]['bottom'].get_yaxis().tick_right()
    axs[0]['top'].set_ylabel('State')
    axs[0]['bottom'].set_ylabel('Action')
    
    fig.legend(handles = handles, loc='lower center', bbox_to_anchor=(.5, .0),
          ncol=5, fancybox=True, shadow=True)
    fig.align_ylabels()
    plt.grid(True)

    plt.show()

def plot_filters():
    descriptions = { "Unfiltered" : ""
                   , "Butterworth": " ($3^{rd}$ order)"
                   , "Median": " ($5^{th}$ order)"
                   , "FIR": " ($11^{th}$ order)"
                   , "EMA": " ($\\alpha=0.6$)"
                   }
    folder = "data/toy/filter_data"
    f, ax = plt.subplots(2, len(descriptions), sharex=True, sharey='row')
    setpoints = pickle.load(open(os.path.join(folder,"setpoints/setpoints.p"), "rb"))
    for i, name in enumerate(descriptions.keys()):
        (outputs, feedback) = pickle.load(open(os.path.join(folder, name+".p"), "rb"))
        ax[0,i].plot(setpoints, label="Target", linewidth=1.5, color="black")
        ax[0,i].plot(feedback, label="Actual", color=utils.theme['purple'])
        ax[0,i].set_yticks([])
        ax[0,i].set_title(name+descriptions[name])
        ax[1,i].plot(outputs, label="Action", color=utils.theme['blue'])
        ax[1,i].set_yticks([0])
        ax[1,i].set_yticklabels([''])
        ax[1,i].set_xlabel('Time')
        ax[1,i].set_xticks([])
        ax[1,i].grid()
    ax[0,0].legend(loc="upper center",fancybox=True, shadow=True, ncol=2, columnspacing=0.8)
    # ax[1,0].legend(loc="upper center",fancybox=True, shadow=True)
    ax[0,0].set_ylabel("State")
    ax[1,0].set_ylabel("Action")
    ax[1,0].set_yticklabels(['0'])
    f.align_ylabels()
    # ax[1,-1].legend(loc='center right',fancybox=True, shadow=True, bbox_to_anchor=(1.7, 0.5))
    # ax[0,0].set_ylabel("State")
    # ax[1,0].set_ylabel("Actions")
    plt.show()


if __name__ == "__main__":
    import sys
    plot_filters()
    # plot_perlin_vs_step()
