import pickle
import matplotlib.pyplot as plt
import numpy as np
import utils
import os


def plot_following(files, ax_top, ax_bottom, plus_caps=None):
    # colors = ['#e41a1c', '#265285', '#4daf4a', '#984ea3', '#6e0178', '#ff7f00']
    colors = utils.colors
    handles = []
    for i, file in enumerate(files):
        (feedback_list, setpoint_list, error_list, time_list, outputs, alg_name) = pickle.load(open(file, "rb"))
        feedback_list = feedback_list[:75]
        setpoint_list = setpoint_list[:75]
        error_list = error_list[:75]
        time_list = time_list[:75]
        outputs = outputs[:75]
        print(len(feedback_list))
        print(len(time_list))
        if i == 0:
            handles.append(ax_top.plot(time_list, setpoint_list, color='black', label='Target')[0])
        line = ax_top.plot(time_list, feedback_list, color=colors[i], alpha=0.8, label=alg_name+ (plus_caps if plus_caps else ""), linestyle=utils.line_styles[i], linewidth=0.8)
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
    # folder = "data/toy/filter_data"
    # folder = "data/toy/pid_critical_damping"
    folder = "data/toy/filter_data_crazy"
    f, ax = plt.subplots(2, len(descriptions), sharex=True)
    setpoints = pickle.load(open(os.path.join(folder,"setpoints/setpoints.p"), "rb"))
    for i, name in enumerate(descriptions.keys()):
        (outputs, feedback) = pickle.load(open(os.path.join(folder, name+".p"), "rb"))
        ax[0,i].plot(setpoints, label="Target", linewidth=1.5, color="black")
        ax[0,i].plot(feedback, label="Actual", color=utils.theme['purple'])
        ax[0,i].set_yticks([])
        ax[0,i].set_title(name+descriptions[name])
        if name != "FIR":
            ax[0,i].set_ylim([-11,-8.9])
        ax[1,i].set_ylim([-1.2,1.2])
        ax[1,i].plot(outputs, label="Action", color=utils.theme['blue'])
        ax[1,i].set_yticks([0])
        ax[1,i].set_yticklabels([''])
        ax[1,i].set_xlabel('Time')
        ax[1,i].set_xticks([])
        ax[1,i].grid()
    # ax[0,0].legend(loc="upper center",fancybox=True, shadow=True, ncol=2, columnspacing=0.8)
    # ax[1,0].legend(loc="upper center",fancybox=True, shadow=True)
    ax[0,0].set_ylabel("State")
    ax[1,0].set_ylabel("Action")
    ax[1,0].set_yticklabels(['0'])
    f.align_ylabels()
    # ax[1,-1].legend(loc='center right',fancybox=True, shadow=True, bbox_to_anchor=(1.7, 0.5))
    # ax[0,0].set_ylabel("State")
    # ax[1,0].set_ylabel("Actions")
    plt.show()
# -9.5


def state_action_distribution(ax, f, outputs, error_list):
    outputs = np.array(outputs)[:,0]
    error_list = np.array(error_list)
    print(outputs.shape, error_list.shape)
    H, yedges, xedges = np.histogram2d(outputs,error_list, bins=(60, 60))
    logged = np.log(1+H)
    print(logged.shape)
    colors = ax.imshow(logged/np.max(logged), interpolation='nearest', origin='low',extent=[-1, 1, -1, 1], aspect='auto', cmap=plt.cm.BuPu)

    ax.plot(np.linspace(-1.,1.,10), np.linspace(-1.,1.,10), color="green", linestyle="--")
    ax.plot([-1,0,0,1], [-0.99,-0.99,0.99,0.99], color="red", linestyle="--", linewidth=2, alpha=0.6)
    # plt.colorbar(colors)
    ax.set_xlabel('State')
    ax.set_ylabel('Action')

    # f.suptitle('Agent state-action distribution')



def plot_state_action():
    folder = "data/toy/state_action"
    file = "TD3_perlin"
    (feedback_list, setpoint_list, error_list, time_list, outputs, alg_name) = pickle.load(open(os.path.join(folder, file+".p"), "rb"))
    fig, ax = plt.subplots(1,1,  figsize=(3.3, 2.3))
    plt.subplots_adjust(left=0.2, right=0.84, bottom=0.2, top=0.97)
    state_action_distribution(ax, fig, outputs, error_list)
    # fig.align_ylabels()


    plt.savefig(os.path.join(folder, file+"_state_action.pdf"))
    plt.show()

def state_action_distribution_reg(ax, f, outputs, error_list):
    outputs = np.array(outputs)[:,0]
    error_list = np.array(error_list)
    print(outputs.shape, error_list.shape)
    H, yedges, xedges = np.histogram2d(outputs,error_list, bins=(60, 60))
    logged = np.log(1+H)
    print(logged.shape)
    colors = ax.imshow(logged/np.max(logged), interpolation='nearest', origin='low',extent=[-1, 1, -1, 1], aspect='auto', cmap=plt.cm.BuPu)

    ax.plot(np.linspace(-1.,1.,10), np.linspace(-1.,1.,10), color="green", linestyle="--")
    ax.plot([-1,0,0,1], [-0.99,-0.99,0.99,0.99], color="red", linestyle="--", linewidth=2, alpha=0.6)
    plt.colorbar(colors)
    ax.set_xlabel('State')
    # ax.set_ylabel('Action')
    ax.set_yticklabels('')

    # f.suptitle('Agent state-action distribution')

def plot_state_action_reg():
    folder = "data/toy/state_action"
    file = "TD3_perlin_reg"
    (feedback_list, setpoint_list, error_list, time_list, outputs, alg_name) = pickle.load(open(os.path.join(folder, file+".p"), "rb"))
    fig, ax = plt.subplots(1,1,  figsize=(3.3, 2.3))
    plt.subplots_adjust(left=0.2, right=1, bottom=0.2, top=0.97)
    state_action_distribution_reg(ax, fig, outputs, error_list)
    # fig.align_ylabels()


    plt.savefig(os.path.join(folder, file+"_state_action.pdf"))
    plt.show()

def plot_following_single():
    f, ax = plt.subplots(2,1,figsize=(3,4.5))
    filename = "TD3_perlin"
    plot_following([f"data/toy/state_action/{filename}.p"], ax[0], ax[1])
    ax[0].set_title("Without CAPS")
    ax[0].set_ylabel('State')
    ax[0].set_ylim(-3.5,2)
    # ax[0].set_yticklabels('')
    ax[0].set_xticklabels('')
    ax[0].legend(framealpha=0.8)
    ax[1].set_ylim(-1,1)
    # ax[1].set_yticklabels('')
    ax[1].set_ylabel('Action')
    f.align_ylabels()
    plt.subplots_adjust(right=0.97, top=0.95,left=0.25,bottom=0.1, hspace=0.1)
    plt.savefig(filename + "_following.pdf")
    plt.show()

def plot_following_single_reg():
    f, ax = plt.subplots(2,1,figsize=(3,4.5))
    filename = "TD3_perlin_reg"
    plot_following([f"data/toy/state_action/{filename}.p"], ax[0], ax[1], "+CAPS")
    ax[0].set_title("With CAPS")
    # ax[0].set_ylabel('State')
    ax[0].set_ylim(-3.5,2)
    ax[0].set_yticklabels('')
    ax[0].set_xticklabels('')
    ax[0].legend(framealpha=0.8)
    ax[1].set_ylim(-1,1)
    ax[1].set_yticklabels('')
    # ax[1].set_ylabel('Action')
    f.align_ylabels()
    plt.subplots_adjust(right=0.97, top=0.95,left=0.25,bottom=0.1, hspace=0.1)
    plt.savefig(filename + "_following.pdf")
    plt.show()

if __name__ == "__main__":
    import sys
    # plot_state_action()
    # plot_state_action_reg()
    # plot_following_single()
    plot_following_single_reg()
    # plot_perlin_vs_step()
