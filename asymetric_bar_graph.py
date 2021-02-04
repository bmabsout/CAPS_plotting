import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import utils


param_sizes = {
    "Pendulum-v0":    
        {(400, 300): 122201, (256, 256): 67073, (128, 128): 17153, (64, 64): 4481, (32, 32): 1217, (16, 16): 353, (8, 8): 113, (4, 4): 41, (1, 1): 8},
    "Reacher-v2":
        {(400, 300): 125702, (256, 256): 69378, (128, 128): 18306, (64, 64): 5058, (32, 32): 1506, (16, 16): 498, (8, 8): 186, (4, 4): 78, (1, 1): 18},
    "Ant-v2":
        {(400, 300): 167508, (256, 256): 96520, (128, 128): 31880, (64, 64): 11848, (32, 32): 4904, (16, 16): 2200, (8, 8): 1040, (4, 4): 508, (1, 1): 130},
    "HalfCheetah-v2":
        {(400, 300): 129306, (256, 256): 71942, (128, 128): 19590, (64, 64): 5702, (32, 32): 1830, (16, 16): 662, (8, 8): 270, (4, 4): 122, (1, 1): 32},
    "Acrobot-v1":
        {(400, 300): 124003, (256, 256): 68355, (128, 128): 17795, (64, 64): 4803, (32, 32): 1379, (16, 16): 435, (8, 8): 155, (4, 4): 63, (1, 1): 15}
}

data = {
    "Pendulum-v0": {
        "DDPG": {
            "Threshold": -160,
            "Baseline":{"size":(400,300), "reward": -145.56, "std": 10.64},
            "Symmetric":{"size":(16,16), "reward": -150.28, "std": 9.08},
            "Asymmetric":{"size":(4,4), "reward": -158.97, "std": 5.33}
        },
        "TD3":  {
            "Threshold": -160,
            "Baseline":{"size":(400,300), "reward": -152.71, "std": 9.47},
            "Symmetric":{"size":(16,16), "reward": -152.42, "std": 6.65},
            "Asymmetric":{"size":(4,4), "reward": -167.57, "std": 14.53}
        },
        "SAC":  {
            "Threshold": -160,
            "Baseline":{"size":(256,256), "reward": -139.86, "std": 8.29},
            "Symmetric":{"size":(16,16), "reward": -155.32, "std": 11.25},
            "Asymmetric":{"size":(4,4), "reward": -153.86, "std": 10.97}
        },
        "PPO":  {
            "Threshold": -200,
            "Baseline":{"size":(64, 64), "reward": -668.60, "std": 551.85},
            "Symmetric":{"size":(128,128), "reward": -193.55, "std": 40.14},
            "Asymmetric":{"size":(128,128), "reward": -193.55, "std": 40.14}
        },
    },
    "Reacher-v2": {
        "DDPG": {
            "Threshold": -6.0,
            "Baseline":{"size":(400,300), "reward": -4.26, "std": 0.25},
            "Symmetric":{"size":(64,64), "reward": -4.75, "std": 0.19},
            "Asymmetric":{"size":(8,8), "reward": -4.80, "std": 0.44}
        },
        "TD3":  {
            "Threshold": -7.0,
            "Baseline":{"size":(400,300), "reward": -6.52, "std": 1.12},
            "Symmetric":{"size":(64,64), "reward": -6.91, "std": 0.74},
            "Asymmetric":{"size":(32,32), "reward": -6.68, "std": 1.21}
        },
        "SAC":  {
            "Threshold": -6.5,
            "Baseline":{"size":(256,256), "reward": -5.96, "std": 0.47},
            "Symmetric":{"size":(128,128), "reward": -6.05, "std": 0.91},
            "Asymmetric":{"size":(16,16), "reward": -6.02, "std": 1.07}
        },
        "PPO":  {
            "Threshold": -5.5,
            "Baseline":{"size":(64, 64), "reward": -4.37, "std": 1.74},
            "Symmetric":{"size":(64,64), "reward": -4.37, "std": 1.74},
            "Asymmetric":{"size":(16,16), "reward": -5.49, "std": 1.00}
        }
    },
    "HalfCheetah-v2": {
        "DDPG": {
            "Threshold": 7000,
            "Baseline":{"size":(400,300), "reward": 7026.01, "std": 202.78},
            "Symmetric":{"size":(64,64), "reward": 7450.01, "std": 950.15},
            "Asymmetric":{"size":(32,32), "reward": 8273.76, "std": 437.66}
        },
        "TD3":  {
            "Threshold": 8000,
            "Baseline":{"size":(400,300), "reward": 8861.92, "std": 870.02},
            "Symmetric":{"size":(64,64), "reward": 8315.13, "std": 262.78},
            "Asymmetric":{"size":(32,32), "reward": 8145.84, "std": 262.55}
        },
        "SAC":  {
            "Threshold": 10000,
            "Baseline":{"size":(256,256), "reward": 11554.76, "std": 779.91},
            "Symmetric":{"size":(64,64), "reward": 10180, "std": 759.10},
            "Asymmetric":{"size":(32,32), "reward": 9619, "std": 158.40}
        },
        "PPO":  {
            "Threshold": 3000,
            "Baseline":{"size":(64, 64), "reward": 3395, "std": 1156.30},
            "Symmetric":{"size":(64, 64), "reward": 3395, "std": 1156.30},
            "Asymmetric":{"size":(32,32), "reward": 3089, "std": 919.25}
        }
    }
}

metadata = {
	"Pendulum-v0": {
		"ylim": [-200, -130]	
	},
	"Reacher-v2": {
		"ylim": [-10, 0]
	},
	"HalfCheetah-v2": {
		"ylim": [0, 12000]
	}
}

env = "Pendulum-v0"

algs_dict = data[env]

ind = np.arange(len(algs_dict))  # the x locations for the groups
width = 0.3  # the width of the bars

# fig, ax = plt.subplots(figsize=(8.1,2.2))
fig, ax = plt.subplots(2,len(data), sharex=True, figsize=(7,5.1))
ax2 = ax[1,0]
ax1 = ax[0,0]
left_color = utils.colors[0]
right_color = utils.colors[1]

# left_color2 = "#26889c"
# right_color2 = "#a0260d"

bottom = -200

sym_param_leg = None
sym_reward_leg = None
asym_param_leg = None
asym_reward_leg = None

closeness = 3/4

for j, env in enumerate(data.keys()):
    algs_dict = data[env]
    ylim = metadata[env]["ylim"]
    ax[1,j].set_ylim(ylim)
    ax[0,j].set_title(env)
    for i, info in enumerate(algs_dict.values()):
        symmetric_param_count = param_sizes[env][info["Symmetric"]["size"]]
        sym_param_leg, = ax[0,j].bar(i - width/2, 100, width=closeness * width, color=left_color, label="symmetric params")
        
        asymmetric_param_count = param_sizes[env][info["Asymmetric"]["size"]]
        asym_param_leg, = ax[0,j].bar(i + width/2, 100*asymmetric_param_count/symmetric_param_count, width=closeness*width, color=right_color, label="asymmetric params")


        sym_reward_leg, = ax[1,j].bar(i -width/2,
            height=info["Symmetric"]["reward"] - ylim[0],
            bottom=ylim[0],
            width=closeness*width,
            yerr=info["Symmetric"]["std"],
            color=left_color,
            label="symmetric rewards"
        )

        asym_reward_leg, = ax[1,j].bar(i + width/2,
            height=info["Asymmetric"]["reward"] - ylim[0],
            bottom=ylim[0],
            width=closeness*width,
            yerr=info["Asymmetric"]["std"],
            color=right_color,
            label="Asymmetric rewards"
        )



# Add some text for labels, title and custom x-axis tick labels, etc.
ax1.set_ylabel('Weight count (%)')
ax1.set_xticks(ind)

# ax2.spines['left'].set_color(left_color)
# ax1.tick_params(axis='y', colors=left_color)
# ax1.yaxis.label.set_color(left_color)
ax1.set_xticklabels(tuple(algs_dict.keys()))
# ax.legend()

ax2.set_ylabel('Rewards')
# ax2.set(ylim=[bottom, -100])
# ax2.spines['right'].set_color(right_color)
# ax2.tick_params(axis='y', colors=right_color)
# ax2.yaxis.label.set_color(right_color)


ax[1,2].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
fig.tight_layout()
fig.align_ylabels()
legend = [
    (sym_param_leg, "symmetric"),
    (asym_param_leg, "asymmetric"),
]

ax[1,1].legend(*zip(*legend), loc='upper center',
          ncol=1)
plt.subplots_adjust(left=0.1, bottom=0.05, right=0.996, wspace=0.3, top=.95, hspace=0.17)
plt.savefig("plots/asymmetric/asymmetric_vs_symmetric.pdf")
# plt.show()
