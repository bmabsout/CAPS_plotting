import os
import csv
import numpy as np
from matplotlib import pyplot as plt

import utils

def bfcalc(rcCommand, rcRate=1., expo=0., superRate=0.7):
    clamp = lambda n, minn, maxn: max(min(maxn, n), minn)
    absRcCommand = abs(rcCommand)

    if rcRate > 2.0:
        rcRate = rcRate + (14.54 * (rcRate - 2.0))
    if expo != 0:
        rcCommand = rcCommand * abs(rcCommand)**3  * expo + rcCommand * (1.0 - expo)
    angleRate = 200.0 * rcRate * rcCommand;
    if superRate != 0:
        rcSuperFactor = 1.0 / (clamp(1.0 - (absRcCommand * (superRate)), 0.01, 1.00))
        angleRate *= rcSuperFactor
    return angleRate


def processLog(fname, rows_taken=None, start_from=0):
    motor_vals = []
    rc_commands = []
    gyro = []
    t = []
    amps = []
    count = 0
    start_read = False

    with open(fname) as f:
        reader = csv.reader(f)
        for row in reader:
            if not start_read:
                if row[0] == 'loopIteration':
                    start_read = True
                    rc_cidx = None
                    for i in range(len(row)):
                        if row[i] == 'time':
                            t_cidx = i
                        if row[i] == 'rcCommand[0]' and rc_cidx is None:
                            rc_cidx = i
                        if row[i] == 'gyroADC[0]':
                            g_cidx = i
                        if row[i] == 'motor[0]':
                            m_cidx = i
                        if row[i] == 'amperageLatest':
                            amp_idx = i
                    continue
            else:
                if count > start_from:
                    amps.append(float(row[amp_idx])/100)
                    t.append(float(row[t_cidx]))
                    rc_commands.append([bfcalc(float(row[rc_cidx])/500),bfcalc(float(row[rc_cidx+1])/500),bfcalc(float(row[rc_cidx+2])/500),bfcalc(float(row[rc_cidx+3])/500)])
                    # rc_commands.append([float(row[rc_cidx]),float(row[rc_cidx+1]),float(row[rc_cidx+2]),float(row[rc_cidx+3])])
                    gyro.append([float(row[g_cidx]), float(row[g_cidx+1]), float(row[g_cidx+2])])
                    motor_vals.append([float(row[m_cidx]), float(row[m_cidx+1]), float(row[m_cidx+2]), float(row[m_cidx+3])])
                if rows_taken and count > rows_taken + start_from:
                    break
                count+=1
    amps = np.array(amps)
    print(fname, count)
    t = np.array(t)/1e6
    avg_t_diff = 0.00137

    motor_vals = 2*np.array(motor_vals)/1600 - 1.
    amplitudess = []
    smoothnesses = []
    for i in range(motor_vals.shape[1]):
        freqs, amplitudes = utils.fourier_transform(motor_vals[:, i], avg_t_diff)
        smoothnesses.append(utils.smoothness(amplitudes))
        amplitudess.append(amplitudes)


    return {'motor_vals':motor_vals, 'rc_commands':rc_commands, 'gyro':gyro, 't':t, 'amplitudess': amplitudess, 'freqs': freqs, 'smoothnesses': smoothnesses, 'amps': amps}

def folder_to_array_dict(folder, rows_taken, start_from=0):
    logs = []
    for file in os.listdir(folder):
        if file.endswith('.csv'):
            fname = os.path.join(folder, file)
            logs.append(processLog(fname, rows_taken, start_from))

    return utils.dict_elems_apply(np.array, utils.dicts_list_to_list_dicts(logs))


def plot_avg_fourier(f, ax, vals):
    utils.plot_fourier( ax
                      , freqs          = vals['freqs'][0]
                      , amplitudes     = vals['amplitudess'].mean(axis=(0,1))
                      , amplitudes_std = vals['amplitudess'].std(axis=(0,1)))
    ax.set_ylim([0,0.04])
    ax.set_xlabel('Frequency (Hz)')
    # ax.set_title(f"Smoothness: {np.mean(vals['smoothnesses'])*10**3:.2f}$\\pm${np.std(vals['smoothnesses'])*10**3:.2f}")

    rc_commands = vals['rc_commands'][:,:,:3]
    print("smoothness:", np.mean(vals['smoothnesses']))
    print("smoothness_std:", np.std(vals['smoothnesses']))
    print("MAE:", np.average(np.abs(rc_commands - vals['gyro'])))
    print("MAES:", np.average(np.abs(rc_commands - vals['gyro']), axis=(1,2)))
    print("MAE_std:", np.std(np.abs(rc_commands - vals['gyro'])))
    print("Amps:", np.average(vals['amps']))
    print("Amps_std:", np.std(vals['amps']))
    # print("RMSE:", np.sqrt(np.average((rc_commands - vals['gyro'])**2)))


import matplotlib.patheffects as path_effects
def plot_motors(f, ax, label, vals):
    percentage_motors = 100*(vals["motor_vals"]/2+0.5)
    first_flight_motors = percentage_motors[0]
    t = vals["t"][0]
    t -= t[0]
    for i in range(first_flight_motors.shape[1]):

        ax.plot(t, first_flight_motors[:,i], label=f"Motor {i+1}", color=utils.colors[i],linewidth=0.7, alpha=1.0/(1+i*0.2), linestyle=utils.line_styles[i])
                # , path_effects=[path_effects.SimpleLineShadow((1.2,-1.2)), path_effects.Normal()])
    if label:
        ax.set_title(label)
    # ax.set_xlim([0.01,0.99])
    # ax.set_xticklabels([])
    ax.set_xlabel('Time (s)')

def fourier_vs_motors_plot(labels=[ "PID","Neuroflight", "PPO+Temporal", "PPO+Spatial", "PPO+CAPS"]):
    f, ax = plt.subplots(2,len(labels),figsize=(7,3.2), sharey='row', sharex=False, constrained_layout=True)
    
    for i, label in enumerate(labels):
        fdir = "./data/reality/" + label
        vals = folder_to_array_dict(fdir, rows_taken=6000, start_from= 3000)
        plot_motors(f, ax[0,i], label, vals)
        plot_avg_fourier(f, ax[1,i], vals)
    ax[0,0].set_ylabel('Motor usage %')
    # ax[0,-1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[0,-1].legend(loc='center right')
    ax[1,0].set_ylabel('Normalized Amplitude')
    # ax[1,-1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[1,-1].legend(loc='center right')

    f.align_ylabels()

    plt.savefig(f"plots/reality/{str(labels)}.pdf")
    plt.show()

def caps_motors_plot():
    f, ax = plt.subplots(1,1)
    vals = folder_to_array_dict("./data/reality/PPO+CAPS", rows_taken=1000, start_from=5000)
    print(utils.dict_elems_apply(np.shape, vals))
    vals["motor_vals"] = vals["motor_vals"][:,:,0:1]
    plot_motors(f, ax, None, vals)
    # ax.legend(loc="upper center", ncol=5, fancybox=True, shadow=True)
    ax.set_ylim([20, 60])
    # ax.set_xticks([])
    ax.set_xlabel('Time(s)')
    # ax.set_yticks([])
    ax.set_ylabel('Motor usage %')

    # plt.axis('off')
    plt.show()

def neuroflight_motors_plot():
    f, ax = plt.subplots(1,1)
    vals = folder_to_array_dict("./data/reality/Neuroflight", rows_taken=1000, start_from=5000)
    vals["motor_vals"] = vals["motor_vals"][:,:,0:1]
    plot_motors(f, ax, None, vals)
    ax.set_ylim([20, 60])
    ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_ylabel('')
    ax.set_ylabel('Motor usage %')

    ax.set_xlabel('')
    # plt.axis('off')
    plt.show()


def fourier_vs_motors_real_plot():
    labels = [ "PID","Neuroflight", "RE+AL" ]
    f, ax = plt.subplots(2,len(labels),figsize=(5,2), sharey='row', sharex=False)
    
    for i, label in enumerate(labels):
        fdir = "./data/reality/" + label
        vals = folder_to_array_dict(fdir, rows_taken=1000, start_from=5000)
        plot_motors(f, ax[0,i], label, vals)
        plot_avg_fourier(f, ax[1,i], vals)
    ax[0,0].set_ylabel('Motor usage %')
    ax[0,-1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[1,0].set_ylabel('Normalized Amplitude')
    ax[1,-1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    f.align_ylabels()

    plt.show()


def plot_following(desired_rpy, rpy, motor_outputs, t_diff=0.00137):
    print(desired_rpy.shape)
    print(rpy.shape)
    print(motor_outputs.shape)
    rpy_std = np.std(rpy[2:,:,:],axis=0)

    f, ax = plt.subplots(4,1, sharex=True, sharey=False, constrained_layout=True, figsize=(3.7,5))

    # plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    t = np.array(range(len(desired_rpy[:,0]))) * t_diff
    # colors = ['#e41a1c', '#265285', '#4daf4a', '#984ea3', '#6e0178', '#ff7f00']

    def plot_desired_vs_actual(i,label):
        desired = ax[i].plot(t, desired_rpy[:,i],"-", label="Desired", linewidth=3, alpha=1, color="black")
        rpy_mean = np.mean(rpy[0:,:,i],axis=0)
        ours=ax[i].plot(t, rpy_mean, linewidth=1.5, alpha=1, linestyle="-", color=utils.colors[2])
        ax[i].fill_between(t, rpy_mean-rpy_std[:,i], rpy_mean+rpy_std[:,i], color=utils.colors[2],alpha=0.5)
        between = ax[i].fill(np.NaN, np.NaN, alpha=0.5, color = utils.colors[2])
        ax[i].set_ylabel(label)
        if i == 0:
            ax[0].legend([desired[0],ours[0]], ["Desired", "Actual"], loc="center left")
        print(rpy.shape)
        print(desired_rpy.shape)
        print(i)
        mae = np.mean(np.abs(rpy[0,:,i] - desired_rpy[:,i]))
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
        ax[i].set_ylabel(label)
        lines = ["-", "-.", ":", "--"]
        for j in range(4):
            ax[i].plot(t, motor_outputs[0, :, j], label="Motor {}".format(j+1), linestyle=lines[j], alpha=0.8)

    plot_motor_outputs(3, "Adapted (%)")
    ax[3].legend(loc="center left")
    f.align_ylabels()
    ax[3].set_xlabel("Time (s)")

    return f, ax


def plot_error_and_motor_outputs(desired_rpy, rpy, motor_outputs, t_diff=0.00137, figsize=(7,2)):
    print(desired_rpy.shape, rpy.shape, motor_outputs.shape)
    f, ax = plt.subplots(1,2, sharex=True, sharey=False, constrained_layout=True, figsize=figsize)

    # plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    t = np.array(range(len(desired_rpy[:,0]))) * t_diff
    # colors = ['#e41a1c', '#265285', '#4daf4a', '#984ea3', '#6e0178', '#ff7f00']

    def plot_error():
        errors = ax[0].plot(t, np.mean(np.abs(desired_rpy - rpy), axis=1),"-", label="Desired", linewidth=3, alpha=1, color="black")
        # ax[0].legend([errors[0]], ["MAE"], loc="upper left")
        ax[0].set_ylabel("MAE (deg/s)")
        ax[0].set_xlabel("Time (s)")

    plot_error()

    def plot_motor_outputs():
        ax[1].set_ylabel("Motor Usage (%)")
        ax[1].set_xlabel("Time (s)")
        lines = ["-", "-.", ":", "--"]
        for j in range(4):
            ax[1].plot(t, motor_outputs[:, j], label="Motor {}".format(j+1), linestyle=lines[j], alpha=0.8)
        ax[1].legend(loc="upper left")

    plot_motor_outputs()
    return f, ax

def DDPG_with_Q_adapt_plot():
    vals = folder_to_array_dict("./data/reality/DDPGxQRegAdapt", rows_taken=350, start_from=1000)
    # vals["motor_vals"] = vals["motor_vals"][:,:,0:1]
    f, ax = plot_following(np.squeeze(vals["rc_commands"])[:,:3], vals["gyro"], vals["motor_vals"])
    plt.savefig("plots/reality/adaptation/DDPGxQRegAdapt.pdf")
    plt.show()


def DDPG_with_Q_adapt_error_plot():
    vals = folder_to_array_dict("./data/reality/DDPGxQRegAdapt", rows_taken=350, start_from=1000)
    # vals["motor_vals"] = vals["motor_vals"][:,:,0:1]
    f, ax = plot_error_and_motor_outputs(vals["rc_commands"][0,:,:3], np.squeeze(vals["gyro"]), np.squeeze(vals["motor_vals"]), figsize=(7,1.6))
    plt.savefig("plots/reality/adaptation/DDPGxQRegAdaptErrors.pdf")
    plt.show()


if __name__ == "__main__":
    # DDPG_with_Q_adapt_plot()
    # DDPG_with_Q_adapt_error_plot()
    # fourier_vs_motors_real_plot()
    # fourier_vs_motors_plot()
    fourier_vs_motors_plot(["Before adaptation", "After adaptation"])
    # fourier_vs_motors_plot(["Before adaptation", "After adaptation"])
    # caps_motors_plot()
    # real_motors_plot()
    # neuroflight_motors_plot()
