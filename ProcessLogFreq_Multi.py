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


def processLog(fname, rows_taken):
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
                t.append(float(row[t_cidx]))
                rc_commands.append([bfcalc(float(row[rc_cidx])/500),bfcalc(float(row[rc_cidx+1])/500),bfcalc(float(row[rc_cidx+2])/500),bfcalc(float(row[rc_cidx+3])/500)])
                # rc_commands.append([float(row[rc_cidx]),float(row[rc_cidx+1]),float(row[rc_cidx+2]),float(row[rc_cidx+3])])
                gyro.append([float(row[g_cidx]), float(row[g_cidx+1]), float(row[g_cidx+2])])
                motor_vals.append([float(row[m_cidx]), float(row[m_cidx+1]), float(row[m_cidx+2]), float(row[m_cidx+3])])
                if count > rows_taken:
                    break
                count+=1

    t = np.array(t)/1e6
    avg_t_diff = 0.00137

    motor_vals = 2*np.array(motor_vals)/1000 - 1.
    amplitudess = []
    smoothnesses = []
    for i in range(motor_vals.shape[1]):
        freqs, amplitudes = utils.fourier_transform(motor_vals[:, i], avg_t_diff)
        smoothnesses.append(utils.smoothness(amplitudes))
        amplitudess.append(amplitudes)


    return {'motor_vals':motor_vals, 'rc_commands':rc_commands, 'gyro':gyro, 't':t, 'amplitudess': amplitudess, 'freqs': freqs, 'smoothnesses': smoothnesses}

def folder_to_array_dict(folder, rows_taken):
    logs = []
    for file in os.listdir(folder):
        if file.endswith('.csv'):
            fname = os.path.join(folder, file)
            logs.append(processLog(fname, rows_taken))

    return utils.dict_elems_apply(np.array, utils.dicts_list_to_list_dicts(logs))


def plot_avg_fourier(f, ax, label, vals):
    utils.plot_fourier( ax
                      , freqs          = vals['freqs'][0]
                      , amplitudes     = vals['amplitudess'].mean(axis=(0,1))
                      , amplitudes_std = vals['amplitudess'].std(axis=(0,1)))
    ax.set_ylim([0,0.08])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_title(f"{label}, smoothness: {np.mean(vals['smoothnesses'])*10**3:.2f}$\\pm${np.std(vals['smoothnesses'])*10**3:.2f}")

    rc_commands = vals['rc_commands'][:,:,:3]

    print("MAE:", np.average(np.abs(rc_commands - vals['gyro'])))
    print("RMSE:", np.sqrt(np.average((rc_commands - vals['gyro'])**2)))



if __name__ == "__main__":
    labels = [ "Neuroflight", "PID", "PPO+CAPS" ]
    f, ax = plt.subplots(1,len(labels),figsize=(5,2), sharey=True, sharex=False)
    
    for i, label in enumerate(labels):
        fdir = "./data/reality/" + label
        vals = folder_to_array_dict(fdir, rows_taken=6000)
        plot_avg_fourier(f, ax[i], label, vals)

    ax[0].set_ylabel('Normalized Amplitude')

    plt.show()
