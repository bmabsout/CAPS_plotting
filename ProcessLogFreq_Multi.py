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


def processLog(fname, rows_taken, start_from=0):
    motor_vals = []
    rc_commands = []
    gyro = []
    t = []
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
                    continue
            else:
                if count > start_from:
                    t.append(float(row[t_cidx]))
                    rc_commands.append([bfcalc(float(row[rc_cidx])/500),bfcalc(float(row[rc_cidx+1])/500),bfcalc(float(row[rc_cidx+2])/500),bfcalc(float(row[rc_cidx+3])/500)])
                    # rc_commands.append([float(row[rc_cidx]),float(row[rc_cidx+1]),float(row[rc_cidx+2]),float(row[rc_cidx+3])])
                    gyro.append([float(row[g_cidx]), float(row[g_cidx+1]), float(row[g_cidx+2])])
                    motor_vals.append([float(row[m_cidx]), float(row[m_cidx+1]), float(row[m_cidx+2]), float(row[m_cidx+3])])
                if count > rows_taken + start_from:
                    break
                count+=1
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


    return {'motor_vals':motor_vals, 'rc_commands':rc_commands, 'gyro':gyro, 't':t, 'amplitudess': amplitudess, 'freqs': freqs, 'smoothnesses': smoothnesses}

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
    ax.set_ylim([0,0.07])
    ax.set_xlabel('Frequency (Hz)')
    # ax.set_title(f"Smoothness: {np.mean(vals['smoothnesses'])*10**3:.2f}$\\pm${np.std(vals['smoothnesses'])*10**3:.2f}")

    rc_commands = vals['rc_commands'][:,:,:3]

    print("MAE:", np.average(np.abs(rc_commands - vals['gyro'])))
    print("RMSE:", np.sqrt(np.average((rc_commands - vals['gyro'])**2)))


import matplotlib.patheffects as path_effects
def plot_motors(f, ax, label, vals):
    percentage_motors = 100*(vals["motor_vals"]/2+0.5)
    first_flight_motors = percentage_motors[0]
    t = vals["t"][0]
    t -= t[0]
    for i in range(first_flight_motors.shape[1]):

        ax.plot(t, first_flight_motors[:,i], label=f"Motor {i+1}", color=utils.colors[i], alpha=0.8)
                # , path_effects=[path_effects.SimpleLineShadow((1.5,-1.5)), path_effects.Normal()])
    if label:
        ax.set_title(label)
    ax.set_xlim([0.01,0.99])
    # ax.set_xticklabels([])
    ax.set_xlabel('Time (s)')

def fourier_vs_motors_plot():
    labels = [ "PID","Neuroflight", "PPO+CAPS" ]
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
    # plt.savefig("plots/reality/fourier_vs_motors.pdf")

def caps_motors_plot():
    f, ax = plt.subplots(1,1)
    vals = folder_to_array_dict("./data/reality/PPO+CAPS", rows_taken=1000, start_from=5000)
    plot_motors(f, ax, None, vals)
    ax.legend()
    ax.set_ylim([20, 60])
    ax.set_xticks([])
    ax.set_xlabel('')
    plt.show()

def neuroflight_motors_plot():
    f, ax = plt.subplots(1,1)
    vals = folder_to_array_dict("./data/reality/Neuroflight", rows_taken=1000, start_from=5000)
    plot_motors(f, ax, None, vals)
    # ax.set_ylim([20, 60])
    ax.set_xticks([])
    ax.set_ylabel('Motor usage %')

    ax.set_xlabel('')
    plt.show()


if __name__ == "__main__":
    fourier_vs_motors_plot()
    # caps_motors_plot()
    # neuroflight_motors_plot()