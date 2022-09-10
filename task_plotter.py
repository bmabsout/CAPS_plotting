import argparse
import numpy as np
import matplotlib   
import matplotlib.pyplot as plt
import os
import math
import sys
import hashlib
from scipy import signal
import pickle
import json

import os
import os.path
class TaskPlotter:
    def __init__(self, flight_log_dirs, flight_log_name=None, labels=[]):
        """
        Args: 
            flight_log_dirs: A list of all directories storing flight logs
            flight_log_name: Name that exists in each flight_log_dirs to compare
            labels: List, in order of flight_log_dirs, of the name of the flight to compare
        """
        self.flight_log_dirs = flight_log_dirs
        self.flight_log_name = flight_log_name
        if len(labels) == 0:
            self.labels = ["Flight-{}".format(i) for i in range(len(flight_log_dirs))]
        else:
            self.labels = labels

        self.unit_rotation = "deg/s"
        self.alpha = 0
        self.colors = ["red", "blue", "green", "magenta"]
        self.markers = ["o", "v", "s"]

        self.T = 0.001 # simulation time step

    def reflect(self, y, yss):
        yscale = 1e6
        y += yscale 
        yss += yscale
        if y[0] > yss: # moving in the positive direction
            print ("HERE")
            yss += (2*(y[0] - yss))
            diff = np.absolute(y - y[0])
            print ("Diff", 2* diff)
            y = y + (2*diff)

        return y, yss

    def damping_ratio(self,ax, actual_rpy, desired_rpy):


        command_times = self.slice_flight(desired_rpy)
        xs = []
        ys = []
        for i in range(len(command_times)):
            if i < len(command_times) - 1:
                start = command_times[i]
                end = command_times[i+1] + 1
                step = actual_rpy[start:end,:]
                ss_step = desired_rpy[start:end,:]
                ss_r = ss_step[0,0]
                ss_p = ss_step[0,1]
                ss_y = ss_step[0,2]
                r = step[:,0]
                p = step[:,1]
                y = step[:,2]




                r, ss_r = self.reflect(r, ss_r) 
                ax2 = ax[0].twinx()
                ax2.plot(range(start, end), r, color='r')

                zeta = self.damping(r, ss_r)
                print ("{} Zeta={:.2f}".format(start, zeta))

                """
                N = len(r) 
                T = self.T
                print ("Step ", start)
                for ax in [r, p, y]:
                    f = np.linspace(0, 1 / T, N)[:N // 2]
                    fft = np.fft.fft(ax)
                    amp = np.abs(fft)[:N // 2] * 1 / N
                    max_a_index = np.argmax(amp)
                    max_a = np.amax(amp)
                    fp = f[max_a_index]
                    level = (1/np.sqrt(2)) *  max_a

                    fu = 0
                    for i in range(max_a_index, len(amp)):
                        if amp[i] >= level:
                            fu = f[i] 
                    fl = 0 
                    if max_a_index > 0:
                        for i in range(max_a_index -1, 0, -1):
                            if amp[i] <= level:
                                fl = f[i]
                    
                    damping_coef = self.half_quadratic_gain(fl, fu, fp)
                    print ("\tZeta={:.2f} fl={} fu={} fp={} Gain={}".format(damping_coef, fl, fu, fp, level))
                """


    def damping(self, y, yss):
        """
        underdamped zeta < 1
        undamped zeta == 0 
        overdamped zeta > 1
        """
        ypeak = np.amax(y)#self.y_peak(y, yss)
        mp = (ypeak - yss) / yss # Maximum overshoot
        print ("ypeak=", ypeak, " mp=", mp)
        return np.sqrt(
            np.power(np.log(mp), 2) / 
            (np.power(np.log(mp), 2) + np.power(math.pi, 2))
        )

    def y_peak(self, y, yss):

        if y[0] <= yss: # moving in the positive direction
            return np.amax(y)
        else:  # moving in the negative direction
            return np.amin(y)

        """
        ypeak = 0
        for y_ in y:
            if y_ > ypeak:
                ypeak = y_
        """


    def half_quadratic_gain(self, fl, fu, fp):

        a = 4 * np.power((fu - fl)/fp, 2)
        b = np.power((fu - fl)/fp, 4)
        c = 0.5 - np.sqrt(np.power(4 + a - b, -1))

        return np.sqrt(c)

    def load_labels(self, filepath):
        labels = []
        with open(filepath) as f:
            first_line = f.readline()
            comment = first_line[1:]
            fields = comment.split(",")
            for f in fields:
                labels.append(f.strip())
        return labels

    def load_data(self, filepath, labels):
        input_size = 0
        for label in labels:
            if label.startswith("x"):
                input_size += 1
        columns = len(labels)
        usecols = list(range(0,columns))
        rs_exists = "rs" in labels
        if rs_exists:
            rs_column = labels.index("rs")
            usecols.pop(rs_column)
        data = np.loadtxt(filepath, dtype='f8', usecols=usecols, delimiter=",", skiprows=1)#[:5000,:]
        reward_size = 1
        action_size = 4
        actual_size = 3
        target_size = 3
        motor_velocity_size = 4
        #input_size = self.input_size 
        dbg_size = data.shape[1] - (reward_size + action_size + actual_size + target_size + motor_velocity_size + input_size)

        start_index = 0
        states = data[:, start_index : start_index + input_size]
        start_index += input_size
        rewards = data[:, start_index]
        start_index += reward_size

        ys = data[:, start_index : start_index + 4]
        start_index += action_size

        actual_rates = data[:, start_index : start_index + 3]
        start_index += actual_size

        target_rates = data[:, start_index : start_index + 3 ]
        start_index += target_size

        motor_velocity = data[:, start_index : start_index + 4 ]
        start_index += motor_velocity_size


        dbg = []
        if dbg_size > 0:
            dbg = data[:, start_index : start_index + dbg_size ]

        if rs_exists:
            rewardsDecomposed = np.loadtxt(filepath, dtype='U', usecols=[rs_column], delimiter=",", skiprows=1)#[:5000,:]
            def jsoner(ix):
                (i, x) = ix
                try:
                    return json.loads(
                        x.replace('"',"")
                        .replace(";",",")
                        .replace(".)", ".0")
                        .replace(".e", ".0e")
                        .replace("array(", "")
                        .replace(")", "")
                        .replace("'", '"')
                    )
                except Exception as e:
                    print(e)
                    print("at row", i)
                    print(rewardsDecomposed[i])
                    exit()
            rewardsDecomposed = list(map(jsoner, enumerate(rewardsDecomposed)))
        else:
            rewardsDecomposed = None
        
        return states, rewards, rewardsDecomposed, ys, actual_rates, target_rates, motor_velocity, dbg

        """
        # All of the setpoints will be of the one that took the longest
        # This is needed if we are comparing episodes that are trying to finish
        # the quickest
        longest_t = t[0] # Start off with the first instance
        ep_index_most_samples = 0
        c = 0
        for ep_times in t:
            if len(ep_times) > len(longest_t):
                longest_t = ep_times
                ep_index_most_samples = c
            c+=1

        end_index = len(longest_t) - 1
        if end > 0:
            end_index = int(end/step_size) - 1
        print ("End=", end, " index=", end_index)

        longest_t = longest_t[:end_index]
        desired_rates = all_desired[ep_index_most_samples][:end_index]
        #threshold = threshold_percent * desired_rates
        desired_r = desired_rates[:,0]
        desired_p = desired_rates[:,1]
        desired_y = desired_rates[:,2]



        if len(motors) > 0:
            motors = motors[:,:end_index,:]
        if len(accel) > 0:
            accel = accel[:,:end_index,:]
        if len(rpms) > 0:
            rpms = rpms[:,:end_index,:]
        """



    def plot_dbg(self, dbg):
        f, ax = plt.subplots(dbg.shape[1], sharex=True, sharey=False)
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        t = np.array(range(dbg.shape[0]))

        for i in range(dbg.shape[1]):
            ax[i].plot(t, dbg[:,i])

    def take_minimum(self, flight_data):
        min_len = min(map(lambda e: e[0].shape[0],flight_data))

        def minimizer(data):
            return tuple(map(lambda e: e[:min_len,], data))
        return list(map(minimizer,flight_data))


    def plot_step_envs(self):
        flight_data = []
        input_index=0
        all_labels = []
        for d in self.flight_log_dirs:
            #filepath = os.path.join(d, self.flight_log_name)
            labels = self.load_labels(d)
            all_labels.append(labels)
            (states, rewards, rewardsDecomposed, actions, actual_rpy, desired_rpy, motor_velocity, dbg) = self.load_data(d, labels)
            flight_data.append(desired_rpy)

        all_desired_rpy = list(zip(*self.take_minimum(flight_data)))
        desired_rpy = np.array(all_desired_rpy)[:,:,0]
        print(desired_rpy.shape)

        f, ax = plt.subplots(desired_rpy.shape[1], sharex=True, sharey=False)
        # f.set_size_inches(10, 5)

        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        t = np.array(range(len(desired_rpy[:,0]))) * 0.001

        def plot_desired_vs_actual(i,label):
            desired = ax[i].plot(t, desired_rpy[:,i], label="desired", linewidth=2, alpha=0.8, color="k")
            ax[i].set_ylabel(label)
        plot_desired_vs_actual(0, "Step")
        plot_desired_vs_actual(1, "Perlin (Training)")
        plot_desired_vs_actual(2, "Perlin (Validation)")
        # plot_desired_vs_actual(2, "Yaw (deg/s)")
        f.align_ylabels(ax)
        return f, ax

    def plot_flights2(self):
        flight_data = []
        input_index=0
        all_labels = []
        for d in self.flight_log_dirs:
            #filepath = os.path.join(d, self.flight_log_name)
            labels = self.load_labels(d)
            all_labels.append(labels)
            (states, rewards, rewardsDecomposed, actions, actual_rpy, desired_rpy, motor_velocity, dbg) = self.load_data(d, labels)
            flight_data.append([actual_rpy, desired_rpy, rewards, actions])

        plot_us = list(zip(*self.take_minimum(flight_data)))
        (all_rpy, all_desired_rpy, all_rewards, all_motor_ouptuts) = plot_us
        desired_rpy = all_desired_rpy[0]
        rpy = np.mean(all_rpy,axis=0)
        rpy_std = np.std(all_rpy,axis=0)
        motor_ouptuts = np.mean(all_motor_ouptuts, axis=0)

        f, ax = plt.subplots(len(plot_us)+1, sharex=True, sharey=False)

        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        t = np.array(range(len(desired_rpy[:,0]))) * 0.001

        print(rpy_std.shape)
        def plot_desired_vs_actual(i,label):
            line = ax[i].plot(t, rpy[:,i], linewidth=1, alpha=1, color="black")
            color = "lightblue"
            between = ax[i].fill_between(t, rpy[:,i]-rpy_std[:,i], rpy[:,i]+rpy_std[:,i], color=color,alpha=0.8)
            desired = ax[i].plot(t, desired_rpy[:,i],"r--", label="desired", linewidth=2, alpha=0.8, color="red")
            p2 = ax[i].fill(np.NaN, np.NaN, alpha=0.8, color = color)
            ax[i].set_ylabel(label)
            ax[i].legend([desired[0], (p2[0], line[0])], ["Desired", "Actual"], loc="right")
        plot_desired_vs_actual(0, "Roll (deg/s)")
        plot_desired_vs_actual(1, "Pitch (deg/s)")
        plot_desired_vs_actual(2, "Yaw (deg/s)")

        ax[3].set_ylabel("Motor Output (\%)")

        m0 = motor_ouptuts[:,0]
        m1 = motor_ouptuts[:,1]
        m2 = motor_ouptuts[:,2]
        m3 = motor_ouptuts[:,3]
        m = [m0, m1, m2, m3]
        
        lines = ["-", "-.", ":", "--"]
        for i in range(4):
            ax[3].plot(t, m[i], label="Motor {}".format(i+1), linestyle=lines[i], alpha=0.8)
        ax[3].legend(loc="right")

        ax[4].plot(t, np.mean(all_rewards,axis=0))

        f.align_ylabels()


        e = np.abs(rpy - desired_rpy)
        t_total = t[-1] - t[0]

        mae = self.mae(e, len(e))
        mse = self.mse(e, len(e))
        iae = self.iae(e)/t_total
        ise = self.ise(e)/t_total
        itae = self.itae(t, e)/t_total
        itse = self.itse(t, e)/t_total
        stds = np.average(rpy_std,axis=0)
        print(stds.shape)
        print ("MAE={:,.2f} {:,.2f} {:,.2f} Ave={:,.2f}".format(*mae, np.average(mae)))
        print ("STD={:,.2f} {:,.2f} {:,.2f} Ave={:,.2f}".format(stds[0],stds[1],stds[2], np.average(stds)))
        print ("MSE={:,.2f} {:,.2f} {:,.2f} Ave={:,.2f}".format(*mse, np.average(mse)))
        print ("IAE={:,.2f} {:,.2f} {:,.2f} Ave={:,.2f}".format(*iae, np.average(iae)))
        print ("ISE={:,.2f} {:,.2f} {:,.2f} Ave={:,.2f}".format(*ise, np.average(ise)))
        print ("ITAE={:,.2f} {:,.2f} {:,.2f} Ave={:,.2f}".format(*itae, np.average(itae)))
        print ("ITSE={:,.2f} {:,.2f} {:,.2f} Ave={:,.2f}".format(*itse, np.average(itse)))
        return f, ax


    def pickle_flights(self, name):
        flight_data = []
        input_index=0
        all_labels = []
        for d in self.flight_log_dirs:
            #filepath = os.path.join(d, self.flight_log_name)
            labels = self.load_labels(d)
            all_labels.append(labels)
            (states, rewards, rewardsDecomposed, actions, actual_rpy, desired_rpy, motor_velocity, dbg) = self.load_data(d, labels)
            flight_data.append([actual_rpy, desired_rpy, rewards, actions])

        (all_rpy, all_desired_rpy, all_rewards, all_motor_ouptuts) = list(zip(*self.take_minimum(flight_data)))
        desired_rpy = all_desired_rpy[0]
        rpy = np.array(all_rpy)
        # rpy_std = np.std(rpy[2:,:,:],axis=0)
        motor_outputs = np.array(all_motor_ouptuts)
        pickle.dump( (desired_rpy, rpy, motor_outputs), open( name+".p", "wb" ) )




    def plot_flights3(self):
        flight_data = []
        input_index=0
        all_labels = []
        for d in self.flight_log_dirs:
            #filepath = os.path.join(d, self.flight_log_name)
            labels = self.load_labels(d)
            all_labels.append(labels)
            (states, rewards, rewardsDecomposed, actions, actual_rpy, desired_rpy, motor_velocity, dbg) = self.load_data(d, labels)
            flight_data.append([actual_rpy, desired_rpy, rewards, actions])

        plot_us = list(zip(*self.take_minimum(flight_data)))
        (all_rpy, all_desired_rpy, all_rewards, all_motor_ouptuts) = plot_us
        desired_rpy = all_desired_rpy[0]
        rpy = np.array(all_rpy)
        print(rpy.shape)
        # rpy_std = np.std(rpy[2:,:,:],axis=0)
        motor_ouptuts = np.array(all_motor_ouptuts)
        print(motor_ouptuts.shape)

        f, ax = plt.subplots(1,1, sharex=True, sharey=False)

        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        t = np.array(range(len(desired_rpy[:,0]))) * 0.001
        colors = ['#e41a1c', '#3673a7', '#4daf4a', '#984ea3', '#6e0178', '#ff7f00']

        def plot_desired_vs_actual(i,label):
            desired = ax[i].plot(t, desired_rpy[:,i],"r--", label="desired", linewidth=2, alpha=1, color=colors[0])
            no_reg = ax[i].plot(t, rpy[0,:,i], linewidth=1.5, alpha=1, label="PPO",linestyle="-.", color=colors[1])
            reg = ax[i].plot(t, rpy[1,:,i], linewidth=1, alpha=1, label="PPO + SRC", color="black")
            # rpy_mean = np.mean(rpy[2:,:,i],axis=0)
            # ours=ax[i].plot(t, rpy_mean, linewidth=1.5, alpha=1, linestyle="-", color=colors[2])
            # ax[i].fill_between(t, rpy_mean-rpy_std[:,i], rpy_mean+rpy_std[:,i], color=colors[2],alpha=0.5)
            # between = ax[i].fill(np.NaN, np.NaN, alpha=0.5, color = colors[2])
            ax[i].set_ylabel(label)
            if i == 0:
                ax[0].legend([desired[0],no_reg[0], reg[0]], ["Desired", "PPO", "PPO + SRC"], loc="right")
            # ax[i].legend([desired[0], (p2[0], line[0])], ["Desired", "Actual (and $\\sigma$)"], loc="right")
        # plot_desired_vs_actual(0, "Roll (deg/s)")
        # plot_desired_vs_actual(1, "Pitch (deg/s)")
        # plot_desired_vs_actual(2, "Yaw (deg/s)")

        ax.set_ylabel("PPO (Motor Usage %)")
        # ax[1].set_ylabel("PPO + SRC (Motor Usage %)")
        ax.yaxis.set_label_position("left")
        ax.yaxis.tick_right()
        # ax[1].yaxis.set_label_position("left")
        # ax[1].yaxis.tick_right()
        colors = [colors[1], colors[4]]
        for i in range(2):
            print(i)
            m0 = motor_ouptuts[i,:,0]*100
            m1 = motor_ouptuts[i,:,1]*100
            m2 = motor_ouptuts[i,:,2]*100
            m3 = motor_ouptuts[i,:,3]*100
            m = [m0]
            
            lines = ["-", "-.", ":", "--"]
            for j in range(len(m)):
                ax.plot(t, m[j], label="Motor {}".format(j+1), linestyle=lines[j], alpha=1, color=colors[i])
        # ax[1].set_ylim(0, 100)
        ax.legend(loc="right")
        f.align_ylabels()
        # ax[1].set_xlabel("Time (s)")

        return f, ax


    def plot_flights(self):


        flight_data = []
        input_index=0
        all_labels = []
        for d in self.flight_log_dirs:
            #filepath = os.path.join(d, self.flight_log_name)
            labels = self.load_labels(d)
            all_labels.append(labels)
            flight_data.append(self.load_data(d, labels))

        flight_data = self.take_minimum(flight_data)
        # Asser targets are the same
        desired_rpy = []
        dbg_len = 0
        for _, _, _, _, _, flight_desired_rpy, _, dbg in flight_data:
            if len(dbg) > 0:
                dbg_len = dbg.shape
                print(dbg_len)
            if len(desired_rpy) > 0:
                # if not np.array([desired_rpy == flight_desired_rpy]).all():
                #     raise SystemExit("Flights do not have same desired rates.")
                print()
            else:
                desired_rpy  = flight_desired_rpy

        if dbg_len == 0:
            num_subplots = 6 
        else:
            num_subplots = 7
        f, ax = plt.subplots(num_subplots, sharex=True, sharey=False)
        f.set_size_inches(10, 5)
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

        # Take the steps then convert to time by step size
        t = np.array(range(len(desired_rpy[:,0]))) * 0.001

        r_ref_label = "$\Omega^*_\phi$"
        p_ref_label = "$\Omega^*_\\theta$"
        y_ref_label = "$\Omega^*_\psi$"
        reflinestyle = "k--"
        error_linestyle = "r--"
        desired_r = desired_rpy[:,0]
        desired_p = desired_rpy[:,1]
        desired_y = desired_rpy[:,2]
        errorbars = False

        ##
        # First plot the reference line for the axis plots
        ##
        if errorbars:
            ax[0].plot(t, desired_r -  threshold , error_linestyle, alpha=0.5)
            ax[0].plot(t, desired_r +  threshold , error_linestyle, alpha=0.5)
        ax[0].plot(t, desired_r, reflinestyle, label=r_ref_label)

        if errorbars:
            ax[1].plot(t, desired_p -  threshold , error_linestyle, alpha=0.5)
            ax[1].plot(t, desired_p +  threshold , error_linestyle, alpha=0.5)
        ax[1].plot(t, desired_p, reflinestyle, label=p_ref_label)

        if errorbars:
            ax[2].plot(t, desired_y -  threshold , error_linestyle, alpha=0.5)
            ax[2].plot(t, desired_y +  threshold , error_linestyle, alpha=0.5)
        ax[2].plot(t, desired_y, reflinestyle, label=y_ref_label)

        
        #####
        # Now plot the collected data
        #####
        alpha = 1
        if len(flight_data) > 1:
            alpha = 0.5

        for data in flight_data:
            labels = all_labels[input_index]
            (states, rewards, actions, actual_rpy, desired_rpy, motor_velocity, dbg) = data 

            accel_xyz = []
            orient_quat = []
            motor_rpy = []
            """
            accel_xyz = states[:,3:6]
            orient_quat = states[:,6:10]
            motor_rpy = states[:,10:14]
            """
            #actions = (np.clip(actions, -1, 1) + 1) * 0.5

            throttle = []
            nn_output = []
            delta = []
            if len(dbg) > 0:
                dbg_start_index = 0
                first_dbg_found = False
                for j in range(len(labels)):
                    if labels[j].startswith("dbg-") and not first_dbg_found:
                        dbg_start_index = j
                        first_dbg_found = True
                    if labels[j] == "dbg-throttle":
                        throttle = dbg[:,j-dbg_start_index]
                    if labels[j] == "dbg-ac-0":
                        index = j - dbg_start_index
                        nn_output = dbg[:,index:index+4]
                    if labels[j] == "dbg-delta-0":
                        index = j - dbg_start_index
                        delta = dbg[:,index:index+4]

            self.plot_flight(ax, 
                  t,
                  self.labels[input_index],
                  self.colors[input_index],
                  alpha,
                  actual_rpy,
                  accel_xyz,
                  orient_quat,
                  motor_rpy,
                  rewards,
                  actions,
                  motor_velocity,
                  throttle = throttle,
                  desired_rpy = desired_rpy,
                  #marker = self.markers[input_index]
                 )

            linestyles = [":", "-", "-.", "--"]
            if len(delta) > 0:
                for i in range(4):
                    ax[-2].plot(t, delta[:,i], linestyle=linestyles[i])#, color=self.colors[input_index])
            if len(nn_output) > 0:
                for i in range(4):
                    ax[-1].plot(t, nn_output[:,i], linestyle=linestyles[i])#, color= self.colors[input_index])
                ax[-1].axhline(1)
                ax[-1].axhline(-1)
                ax[-1].axhline(0.5, linestyle="--", color='k')
                ax[-1].axhline(-0.5,linestyle="--",color='k' )

            #self.waves(ax,t, actions, desired_rpy)
            #self.damping_ratio(ax, actual_rpy, desired_rpy)
            #self.plot_motor_spectrum(ax[4],actual_rpy, desired_rpy, actions)
            #self.plot_full_motor_spectrum(actions)
            #print (self.labels[i])
            metrics = self.performance_metrics(actual_rpy, desired_rpy, actions, motor_velocity)
            print ("\tAverage Error=", metrics[0])
            print ("\tAverage Motor Output=", metrics[1])
            print ("\tAverage Y delta=", metrics[2])

            """
        print ("MAE={:,.2f} {:,.2f} {:,.2f} Ave={:,.2f}".format(*mae, np.average(mae)))
        print ("MSE={:,.2f} {:,.2f} {:,.2f} Ave={:,.2f}".format(*mse, np.average(mse)))
        print ("IAE={:,.2f} {:,.2f} {:,.2f} Ave={:,.2f}".format(*iae, np.average(iae)))
        print ("ISE={:,.2f} {:,.2f} {:,.2f} Ave={:,.2f}".format(*ise, np.average(ise)))
        print ("ITAE={:,.2f} {:,.2f} {:,.2f} Ave={:,.2f}".format(*itae, np.average(itae)))
        print ("ITSE={:,.2f} {:,.2f} {:,.2f} Ave={:,.2f}".format(*itse, np.average(itse)))
            """
            #if len(dbg)>0:
            #self.plot_dbg(dbg)

            input_index+=1

            #self.slice_flight(actual_rpy, desired_rpy)
        for i in range(3):
            ax[i].axhline(y=0, color='k')
            ax[i].grid(True)
            ax[i].legend(loc="right")

        ax[0].set_ylabel("Roll ({})".format(self.unit_rotation))
        ax[1].set_ylabel("Pitch ({})".format(self.unit_rotation))
        ax[2].set_ylabel("Yaw ({})".format(self.unit_rotation))
        # Since we have a shcared X axis, set the last one
        #ax[-1].set_xlabel("Steps (s)")
        ax[-1].set_xlabel("Time (s)")

        #plt.show()
        return f,ax

    def waves(self, axs, t, motor, desired_rpy):
        ax = axs[-3]
        command_times = self.slice_flight(desired_rpy)
        for i in range(len(command_times)):
            if i < len(command_times) - 1:
                start = command_times[i]
                stop = command_times[i+1] + 1

                end = start + int((stop - start)/4)

                step = motor[start:end,:]

                m1 = step[:,0]
                m2 = step[:,1]
                m3 = step[:,2]
                m4 = step[:,3]

                #for m in [m1]:
                m = m1
                #xx = np.linspace(start,end, 10000)
                ts = t[start:end]
                #deg = int((ts[-1] - ts[0]))
                #print("deg=", deg)
                coef = np.polyfit(ts, m, 50 )
                #print ("t=", start , ", ", coef)
                p = np.poly1d(coef)
                ax.plot(ts, p(ts), 'k',linewidth=2)       

    def plot_ref(self, ax, desired_rpy):

        r = desired_rpy[:,0]
        p = desired_rpy[:,1]
        y = desired_rpy[:,2]
        command_times = self.slice_flight(desired_rpy)
        print ("Len R=", len(r))
        print (command_times)
        xs = []
        ys = []
        last_rpy = np.zeros(3)
        for i in range(len(command_times)):
            start = command_times[i]
            if i < len(command_times) - 1:
                end = command_times[i+1] + 1
            else:
                end = len(r)
            #print ("Start time=", start, " End time=", end)
            #print ("Len R=", len(r), "Len P=", len(p), " Len Y=", len(y))
            current_sp = np.array([r[start], p[start], y[start]])
            #print ("SP=", current_sp)

            t = np.linspace(0, (end-start)*0.001)
            print (t)



            tau = 0.1 
            V0 = last_rpy
            for i in range(3):
                A = current_sp[i] 
                print ("AX=", i, " A=", A, " V0=", V0[i])
                
                if V0[i] > A: #Decel
                    tau = 0.1
                    color = 'r'
                else: # Accel
                    tau = 0.187
                    color = 'g'
                ref = V0[i]*np.exp(-t/tau) + A * (1 - np.exp(-t/tau))
                ax[i].plot(start + t/0.001, ref, color=color)
            last_rpy = current_sp.copy()



    def plot_rate_spectrum(self):

        np.set_printoptions(precision=3)
        np.set_printoptions(threshold=sys.maxsize)


        flight_data = []
        i=0
        for d in self.flight_log_dirs:
            filepath = os.path.join(d, self.flight_log_name)
            flight_data.append(self.load_data(filepath))

        f, ax = plt.subplots(3, sharex=True, sharey=False)
        c = 0
        for data in flight_data:
            (states, rewards, rewardsDecomposed, actions, actual_rpy, desired_rpy) = data 

            N = len(actual_rpy) 
            T = self.T

            end = 20
            for i in range(3):
                f = np.linspace(0, 1 / T, N)[:N // 2]
                fft = np.fft.fft(actual_rpy[:,i])
                amp = np.abs(fft)[:N // 2] * 1 / N

                bins = np.linspace(1, 500, 501)
                digitize = np.digitize(f, bins)
                width = f[1] - f[0]
                ax[i].bar(f, amp, width=width, label=self.labels[c], alpha=0.25)
                #print ("f=", f)
                #print ("A=", amp)

                ax[i].legend(loc="right")
                ax[i].set_xlim(left=0, right = 20)
            c+=1

        plt.show()

    def plot_full_motor_spectrum(self, actions):
        f, ax = plt.subplots(1)
        #actions = actions.flatten()
        actions = actions[:,0]
        actions = signal.detrend(actions)
        N = len(actions) 
        print ("Len actions=", N)
        T = self.T
        f = np.linspace(0, 1 / T, N)[:N // 2]
        fft = np.fft.fft(actions)
        amp = np.abs(fft)[:N // 2] * 1 / N
        max_amp = amp[amp>=0.5]
        max_f = f[np.argmax(amp)]
        #print ("M",i, "T=", start, " A Max=", np.amax(amp), " A>0.5=", max_amp, " @ f=", max_f, " f=", f)
        #xs += (start + np.array(list(range(len(amp))))).tolist()
        #ys += amp.tolist()
        ax.bar(f, amp, linewidth=2)



    def plot_motor_spectrum(self, ax, actual_rpy, desired_rpy, actions):

        command_times = self.slice_flight(desired_rpy)
        xs = []
        ys = []
        for i in range(len(command_times)):
            if i < len(command_times) - 1:
                start = command_times[i]
                end = command_times[i+1] + 1
                step = actions[start:end,:]

                m1 = step[:,0]
                m2 = step[:,1]
                m3 = step[:,2]
                m4 = step[:,3]

                N = len(m1) 
                T = self.T
                i = 0
                for m in [m1, m2, m3, m4]:
                #for m in [m1]:
                    f = np.linspace(0, 1 / T, N)[:N // 2]
                    fft = np.fft.fft(m)
                    amp = np.abs(fft)[:N // 2] * 1 / N
                    max_amp = amp[amp>=0.5]
                    max_f = f[np.argmax(amp)]
                    print ("M",i, "T=", start, " A Max=", np.amax(amp), " A>0.5=", max_amp, " @ f=", max_f, " f=", f)
                    #xs += (start + np.array(list(range(len(amp))))).tolist()
                    #ys += amp.tolist()
                    i+= 1
                #ax.axvline(start, linewidth=2, color='r', alpha=0.5)
        #ax.bar(xs, ys, linewidth=2)

    def plot_progress(self, remember=False):

        #f, ax = plt.subplots(dbg.shape[1], sharex=True, sharey=False)
        data_indices = [0,1,2,3]
        ylabels = ["Tracking Error", "Motor Output", "Motor Acceleration", "Total Reward" ]

        num_subplots = len(data_indices)
        f, ax = plt.subplots(num_subplots, sharex=True, sharey=False)

        flight_count = 0

        cache_name = ""
        cache = {}
        for flight_log_dir in self.flight_log_dirs:
            trial_id = os.path.basename(os.path.abspath(os.path.join(flight_log_dir, "..")))
            cache_filename = hashlib.sha256(trial_id.encode()).hexdigest() + ".cache"
            cache_filepath = os.path.join("/tmp", cache_filename)
            print ("Checking for cache at ", cache_filepath)
            if os.path.isfile(cache_filepath):
                #print ("Found cache file for ", trial_id)
                # Cache entry
                # trial_id, step, record data
                #

                # XXX For the first one, numpy things its just a ID array
                # so wait fir the second result
                cache_data = np.loadtxt(cache_filepath)#[:5000,:]
                if len(cache_data.shape) > 1:
                    #print ("Loading cache from ", cache_filepath)
                    for record in cache_data:
                        #id_ = record[0]
                        step = int(record[0])
                        data = record[1:]
                        if not (trial_id in cache):
                            cache[trial_id] = {}
                        if not (step in cache[trial_id]):
                            cache[trial_id][step] = []

                        cache[trial_id][step] = data.tolist()

        all_xs = []
        all_ys = []
        headers = None #["Setpoint", "Change","Action"]
        for flight_log_dir in self.flight_log_dirs:
            x = []
            y = []
            trial_id = os.path.basename(os.path.abspath(os.path.join(flight_log_dir, "..")))

            dir_names = self.get_immediate_subdirectories(flight_log_dir)
            print(trial_id)
            steps = map(lambda dir_name: int(dir_name.split("_")[1]), dir_names)
            steps_sorted = sorted(steps)
            for step in steps_sorted:

                in_cache = False
                #print ("Checking ", trial_id, " is in cache")
                if trial_id in cache:
                    #print (trial_id, " is in cache. Checking step in cache ", step)
                    if step in cache[trial_id]:
                        in_cache = True
                        #print ("\t {} {} in cache".format(trial_id, step))

                if not in_cache:
                        d = os.path.join(flight_log_dir, "ckpt_" + str(step))#, self.flight_log_name)
                        print(os.listdir(d))
                        onlyfiles = [os.path.join(d, f) for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
                        metric_averages = []
                        print (self.labels[flight_count], " Processing ", d)
                        for filepath in onlyfiles:

                            #if os.path.isfile(filepath):
                            (states, rewards, rewards_decomposed, actions, actual_rpy, desired_rpy,
                             motor_velocity, dbg) = self.load_data(filepath, self.load_labels(filepath))
                            # print(rewards_decomposed)
                            if not headers:
                                headers = list(rewards_decomposed[0].keys())
                            print(headers)

                            decomp_rewards = np.average(list(
                                map(lambda dic: [dic[k] for k in headers], rewards_decomposed)
                            ), axis=0)

                            metrics = self.performance_metrics(actual_rpy,
                                                               desired_rpy,
                                                               actions,
                                                               motor_velocity) + [np.average(rewards,axis=0)]+list(decomp_rewards)
                                                              # axis=0).tolist()
                            metric_averages.append(metrics)
                        metrics = np.average(metric_averages, axis=0)

                        print ("\tAverage Error=", metrics[0])
                        print ("\tAverage Motor Output=", metrics[1])
                        print ("\tAverage Motor Acceleration=", metrics[2])
                        # Add to cache so we dont have to do this again next time
                        if not (trial_id in cache):
                            cache[trial_id] = {}
                        cache[trial_id][step] = metrics.tolist()
                        y.append(metrics)
                        x.append(step)
                else:
                    if step == steps_sorted[-1]:
                        print ("{} last step={} in cache.".format(trial_id, step))# cache[trial_id][step] )
                        print(self.flight_log_name)
                        filepath = os.path.join(flight_log_dir, str(step))
                        #print (self.labels[flight_count], " Cached Progessing ", filepath)
                    y.append(cache[trial_id][step])
                    x.append(step)

            #print ("Writing cache ", trial_id)
            try:
                self.write_cache(cache[trial_id], trial_id)
            except:
                pass
            all_xs.append(x)
            all_ys.append(y)


        #print ("X=", x)
        #print ("Y=", y)
        all_xs = np.array(list(zip(*all_xs)))
        print(all_xs.shape)
        all_ys = np.array(list(zip(*all_ys)))
        print(all_ys.shape)
        if remember:
            pickle.dump((all_xs, all_ys, headers),open(remember,"wb"))
        xx = np.mean(all_xs,axis=1)
        yy = np.mean(all_ys,axis=1)
        yystd = np.std(all_ys,axis=1)

        # plot a single trial for all the metrics
        for i in range(len(ax)-1):
            metric_index = data_indices[i]
            data = yy[:,metric_index]
            std = yystd[:,metric_index]
            alpha = 1
            color = "lightblue"
            ax[i].fill_between(xx, data-std, data+std, color=color,alpha=0.8)
            if flight_count == len(self.flight_log_dirs) -1:
                line = ax[i].plot(xx, data , marker='.',  label=str(flight_count), alpha=alpha, color="k")
            else:
                line = ax[i].plot(xx, data , marker='.',  label=str(flight_count), alpha=alpha)

            p2 = ax[i].fill(np.NaN, np.NaN, alpha=0.8, color = color)
            # ax[i].legend([(p2[0], line[0])], ["Mean (and $\\sigma$)"])
            #
            #ax[i].set_yscale('log')

        #ax[4].yscale('log')
        def plot_rewards():
            leg = []
            colors = ["red", "blue", "green", "magenta"]
            for i in range(3, yy.shape[1]):
                print(i)
                data = yy[:,i]
                std = yystd[:,i]
                alpha = 1
                line = ax[len(ax)-1].plot(xx, data , marker='.',  label=str(flight_count), alpha=alpha, color=colors[i-3])
                ax[len(ax)-1].fill_between(xx, data-std, data+std,alpha=0.3,color=colors[i-3])

                p2 = ax[len(ax)-1].fill(np.NaN, np.NaN, alpha=0.8, color=colors[i-3])
                leg.append((p2[0],line[0]))
            # print(leg, headers)
            if headers:
                ax[(len(ax)-1)].legend(leg, headers)
        plot_rewards()


        flight_count += 1


        for i in range(len(ax)):
            metric_index = data_indices[i]
            ax[i].set_ylabel(ylabels[metric_index])
        # Since we have a shcared X axis, set the last one
        
        ax[0].set_yscale('log')
        ax[0].yaxis.set_major_formatter(plt.ScalarFormatter())
        ax[0].set_yticks([200,60,25,10])
        # ax[-1].set_yscale('symlog')
        # ax[-1].yaxis.set_minor_formatter(plt.NullFormatter())
        # ax[-1].set_yticks(ax[-1].get_yticks()[::3])
        # ax[-1].yaxis.set_major_formatter(plt.LogFormatter())
        # ax[-1].yaxis.set_major_formatter(plt.NullFormatter()) 
        # plt.yticks([0.5,0.8,0.9,0.96,0.98, 0.99])
        ax[-1].set_xlabel("Timesteps")
        print(ax[-1].get_yticks())
        f.align_ylabels(ax)
        f.set_size_inches(7,10)
        for i in range(len(self.labels)):
            print ("{}:{}".format(i, self.labels[i]))

        plt.show()

    def read_cache(self):
        pass

    def mae(self, e, n):
        return np.sum(np.abs(e), axis=0)/n

    def mse(self, e, n):
        return np.sum(np.power(e, 2), axis=0)/n


    def iae(self, e):
        return np.sum(np.abs(e), axis=0)

    def ise(self, e):
        return np.sum(np.power(e, 2), axis=0)

    def itae(self, t, e):
        ts = np.transpose(np.array([t,]*3))
        return np.sum(ts * np.abs(e), axis=0)

    def itse(self, t, e):
        ts = np.transpose(np.array([t,]*3))
        return np.sum(ts * np.power(e, 2), axis=0)


    def print_metrics(self):
        """ Take the average metrics from all files in a directory"""
        d = self.flight_log_dirs[0]
        onlyfiles = [os.path.join(d, f) for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
        es = []
        _mae = []
        _mse = []
        _iae = []
        _ise = []
        _itae = []
        _itse = []

        for filepath in onlyfiles:
            (states, rewards, rewardsDecomposed, actions, actual_rpy, desired_rpy,
             motor_velocity, dbg) = self.load_data(filepath, self.load_labels(filepath))
            e = np.abs(actual_rpy - desired_rpy)
            t = np.array(range(len(desired_rpy))) * 0.001
            t_total = t[-1] - t[0]

            mae = self.mae(e, len(e))
            mse = self.mse(e, len(e))
            iae = self.iae(e)/t_total
            ise = self.ise(e)/t_total
            itae = self.itae(t, e)/t_total
            itse = self.itse(t, e)/t_total

            _mae.append(mae)
            _mse.append(mse)
            _iae.append(iae)
            _ise.append(ise)
            _itae.append(itae)
            _itse.append(itse)

        mae = np.average(_mae, axis=0)
        mse = np.average(_mse, axis=0)
        iae = np.average(_iae, axis=0)
        ise = np.average(_ise, axis=0)
        itae = np.average(_itae, axis=0)
        itse = np.average(_itse, axis=0)


        print ("MAE={:,.2f} {:,.2f} {:,.2f} Ave={:,.2f}".format(*mae, np.average(mae)))
        print ("MSE={:,.2f} {:,.2f} {:,.2f} Ave={:,.2f}".format(*mse, np.average(mse)))
        print ("IAE={:,.2f} {:,.2f} {:,.2f} Ave={:,.2f}".format(*iae, np.average(iae)))
        print ("ISE={:,.2f} {:,.2f} {:,.2f} Ave={:,.2f}".format(*ise, np.average(ise)))
        print ("ITAE={:,.2f} {:,.2f} {:,.2f} Ave={:,.2f}".format(*itae, np.average(itae)))
        print ("ITSE={:,.2f} {:,.2f} {:,.2f} Ave={:,.2f}".format(*itse, np.average(itse)))

    def write_cache(self, cache, trial_id):
        data = []
        #print ("Cache=", cache)
        for step, metrics in cache.items():
            data.append([step] + metrics)
        #print ("Data=", data)
        cache_filename = hashlib.sha256(trial_id.encode()).hexdigest() + ".cache"
        cache_filepath = os.path.join("/tmp", cache_filename)

        #print ("Writing to cache...", cache_filepath)
        np.savetxt(cache_filepath, data)

    def get_immediate_subdirectories(self, a_dir):
        return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

    def average_performance(self):

        onlyfiles = [f for f in os.listdir(self.flight_log_dirs) if os.path.isfile(os.path.join(mypath, f))]

        errors = []
        power = []
        for filename in onlyfiles:
            filepath = os.path.join(self.flight_log_dirs, filename)
            states, rewards, rewardsDecomposed, actions, actual_rates, target_rates, motor_velocity = self.load_data(filepath)
            average_error = np.average(np.abs(actual_rpy - desired_rpy))
            errors.append(average_error)
            average_power = np.average(actions)
            power.append(average_power)


        return { "error": np.average(errors), "output": np.average(power)}


    def performance_metrics(self, actual_rpy, desired_rpy, actions, motor_velocity):
        """
        Args:
            actions: Numpy array of all the flight actions each of shape (4,1)

        Returns:
            List of metrics

        """
        error = np.average(np.abs(actual_rpy - desired_rpy), axis=0)
        #print ("\tAverage Error=", error)

        action_total = np.average(actions, axis=0)
        #print ("\tAverage Motor Output=", action_total)

        #print ("\tAverage Motor Velocity=", np.average(motor_velocity))
        command_times = self.slice_flight(desired_rpy)

        """
        rate_oscillations = []
        output_oscillations =[] 
        for i in range(len(command_times)):
            if i < len(command_times) - 1:
                start = command_times[i]
                end = command_times[i+1] + 1
                rate_oscillations.append(self.rate_oscillation(actual_rpy[start:end,:]))
                output_oscillations.append(self.output_oscillation(actions[start:end,:]))


        ave_rate_oscillations = np.average(rate_oscillations)
        ave_output_oscillations = np.average(output_oscillations)
        print ("\tAverage Rate Oscillation=", ave_rate_oscillations)
        print ("\tAverage Motor Output Oscillation=", ave_output_oscillations)
        """

        y_diff = np.average(np.abs(np.diff(actions, axis=0)))
        #print ("\tAverage Y delta=", y_diff)
        e = desired_rpy - actual_rpy

        """
        t_total = t[-1] - t[0]
        mae = self.mae(e, len(e))
        mse = self.mse(e, len(e))
        iae = self.iae(e)/t_total
        ise = self.ise(e)/t_total
        itae = self.itae(t, e)/t_total
        itse = self.itse(t, e)/t_total
        """
 

        return [np.average(error), np.average(action_total), y_diff]

    def slice_flight(self,  desired_rpy):
        """
        Slice the flight into steps

        """

        current_command = None
        command_times = []
        for step in range(len(desired_rpy)):
            if (len(command_times) == 0) or (not (current_command == desired_rpy[step]).all()):
                command_times.append(step)
                current_command = desired_rpy[step]
        return command_times 
        # Now that we have each step we can analyze each slice
        #

    def output_oscillation(self, s):
        """ Return """
        signal = s# * 1000.0
        m1 = signal[:,0]
        m2 = signal[:,1]
        m3 = signal[:,2]
        m4 = signal[:,3]

        N = len(m1) 
        T = self.T
        penalty = 0
        for m in [m1, m2, m3, m4]:
            f = np.linspace(0, 1 / T, N)[:N // 2]
            fft = np.fft.fft(m)
            amp = np.abs(fft)[:N // 2] * 1 / N
            penalty += np.sum(f * amp)
        return penalty


    def rate_oscillation(self, s):
        signal = s 
        r = signal[:,0]
        p = signal[:,1]
        y = signal[:,2]

        N = len(r) 
        T = self.T
        penalty = 0
        for ax in [r,p,y]:
            f = np.linspace(0, 1 / T, N)[:N // 2]
            fft = np.fft.fft(ax)
            amp = np.abs(fft)[:N // 2] * 1 / N
            penalty += np.sum(f * amp)
        return penalty

    def plot_spectrum(self, actual_rpy, desired_rpy):
        N = len(actual_rpy) 
        f = np.linspace(0, 1 / self.T, N)[:N // 2]

        fft = np.fft.fft(actual_rpy[:,0])
        amp = np.abs(fft)[:N // 2] * 1 / N

        """
        plt.figure()
        plt.ylabel("deg/s")
        plt.xlabel('t')
        plt.plot(range(len(actual_rpy[:,0])), actual_rpy[:,0])

        plt.figure()
        plt.ylabel("Amplitude")
        plt.xlabel("Frequency [Hz]")
        #print ("F=", f[:N //2], " A=", len(amp))
        plt.bar(f, amp , width=1.5)  # 1 / N is a normalization factor
        plt.show()
        """


        #print ("start=", start, " end=", end, " Amp=", amp)
        #for i in range(len(amp)):
        #    print ("f=", f[i], " amp=", amp[i])
        oscillation_penalty = np.sum(f * amp)
        print ("Penalty = ", oscillation_penalty)

    def plot_flight(self, ax, 
                  t,
                  label,
                    color,
                  alpha,
                  true_rpy,
                  accel_xyz,
                  orient_quat,
                  motor_rpy,
                  rewards,
                  u,
                  motor_velocity,
                  dbg = None,
                  throttle = [],
                  desired_rpy = [],
                    marker = None
                  ):
        """
        Args:
            ax: An array of axis for the subplot that was created
            u: The control signal
        """

        actual_r = true_rpy[:,0]
        actual_p = true_rpy[:,1]
        actual_y = true_rpy[:,2]

        res_linewidth = 2

        ax[0].plot(t, actual_r, label=label, linewidth=res_linewidth, alpha=alpha, color=color)
        ax[1].plot(t, actual_p, label=label, linewidth=res_linewidth, alpha=alpha, color=color)
        ax[2].plot(t, actual_y, label=label, linewidth=res_linewidth, alpha=alpha, color=color)
        subplot_index = 3


        ## ACCEL 

        """

        if len(accel) > 0:
            ax[subplot_index].set_ylabel("Accel")

            for i in range(len(accel)):
                alpha = 1
                if len(self.alphas) > 0:
                    alpha = self.alphas[i]
                x = accel[i][:,0]
                y = accel[i][:,1]
                z = accel[i][:,2]
                ax[subplot_index].plot(t, x, label="x", linestyle=':', alpha=alpha)
                ax[subplot_index].plot(t, y, label="y", linestyle='-', alpha=alpha)
                ax[subplot_index].plot(t, z, label="z", linestyle='--', alpha=alpha)


            ax[subplot_index].legend(loc="right")
            subplot_index += 1


        motorcolor = [['b','r','#ff8000','#00ff00'], ['m']*4]

        if len(rpms) > 0:
            ax[subplot_index].set_ylabel("RPM")


            for i in range(len(rpms)):
                alpha = 1
                if len(self.alphas) > 0:
                    alpha = self.alphas[i]
                m0 = rpms[i][:,0]
                m1 = rpms[i][:,1]
                m2 = rpms[i][:,2]
                m3 = rpms[i][:,3]
                 
                ax[subplot_index].plot(t, m0, label="{} M1".format(self.labels[i]), linestyle=':', color=colors[i], alpha=alpha)#, color=motorcolor[i][0])
                ax[subplot_index].plot(t, m1, label="{} M2".format(self.labels[i]), linestyle="-", color=colors[i], alpha=alpha)#, color=motorcolor[i][1],)
                ax[subplot_index].plot(t, m2, label="{} M3".format(self.labels[i]), linestyle="-.", color=colors[i], alpha=alpha)#, color=motorcolor[i][2],)
                ax[subplot_index].plot(t, m3, label="{} M4".format(self.labels[i]), linestyle='--', color=colors[i], alpha=alpha)#color=motorcolor[i][3],
                ax[subplot_index].legend( loc='upper right', ncol=4)

            subplot_index += 1

        """
        # Motor values 
        if len(u) > 0:
            #ax[subplot_index].set_ylabel("PWM (\Large$\mu$s)")
            ax[subplot_index].set_ylabel("U (\%)")

            i=0
            #if len(self.alphas) > 0:
            #    alpha = self.alphas[i]
            m0 = u[:,0]
            m1 = u[:,1]
            m2 = u[:,2]
            m3 = u[:,3]
            m = [m0, m1, m2, m3]
            
            lines = ["-", "-.", ":", "--"]
            motor_color = color
            for i in range(4):
                ax[subplot_index].plot(t, m[i], label="{} M{}".format(label, i+1), linestyle=lines[i], alpha=alpha,  marker = marker)#, color=motorcolor[i][0])

            ax[subplot_index].legend( loc='upper right', ncol=4)

            """
            command_times = self.slice_flight(desired_rpy)
            offset = 1750
            for i in range(len(command_times)):
                if i < len(command_times) - 1:
                    start = command_times[i]
                    end = command_times[i+1] + 1

                    xx = np.linspace(start + offset,end, 10000)
                    ts = t[start+offset:end]
                    coef = np.polyfit(ts, m0[start+offset:end], 50)
                    print ("t=", start , ", ", coef)
                    p = np.poly1d(coef)
                    ax[subplot_index].plot(xx, p(xx), 'r',linewidth=2, linestyle=':')       
            """



            subplot_index += 1

        if len(throttle) > 0:
            ax[subplot_index].set_ylabel("Throttle (\%)")
            ax[subplot_index].plot(t,throttle)
            subplot_index += 1


        if len(motor_velocity) > 0:
            ax[subplot_index].set_ylabel("RPM")

            i=0
            #if len(self.alphas) > 0:
            #    alpha = self.alphas[i]
            m0 = motor_velocity[:,0]
            m1 = motor_velocity[:,1]
            m2 = motor_velocity[:,2]
            m3 = motor_velocity[:,3]

            ax[subplot_index].plot(t, m0, label="{} M1".format(self.labels[i]), linestyle=':', alpha=alpha)#, color=motorcolor[i][0])
            ax[subplot_index].plot(t, m1, label="{} M2".format(self.labels[i]), linestyle="-",  alpha=alpha)#, color=motorcolor[i][1],)
            ax[subplot_index].plot(t, m2, label="{} M3".format(self.labels[i]), linestyle="-.", alpha=alpha)#, color=motorcolor[i][2],)
            ax[subplot_index].plot(t, m3, label="{} M4".format(self.labels[i]), linestyle='--', alpha=alpha)#color=motorcolor[i][3],
            ax[subplot_index].legend( loc='upper right', ncol=4)

            subplot_index += 1

        if len(rewards) > 0:
            lR = np.abs(rewards)
            #lR = np.log10(lR)
            lR[rewards < 0] = -lR[rewards < 0]

            ax[subplot_index].set_ylabel("Rewards")
            ax[subplot_index].plot(t, lR)
            ax[subplot_index].set_yscale('linear')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs="+", help='Single file or directory')
    parser.add_argument('--filename', help='Single file or directory')
    parser.add_argument('--labels', help="")
    parser.add_argument('-f', '--plot-flight', action="store_true")
    parser.add_argument('-p', '--plot-progress', action="store_true")
    parser.add_argument('-m', '--metrics', action="store_true")
    parser.add_argument('-r', '--remember', type=str, default=None)
    parser.add_argument('-s', '--save-fig', help='Save plot instead of showing - only enabled for plot flight', action="store_true")

    args = parser.parse_args()
    
    labels = args.labels.split(",") if args.labels else [] 

    plotter = TaskPlotter(args.input, args.filename, labels=labels)
    np.set_printoptions(precision=3)
    np.set_printoptions(threshold=sys.maxsize)

    #training_run = args.filename.split("_")[-1].split("/")[0]
    checkpoint_num = args.input[0].split("/")[-2]
    trial_num = args.input[0].split("/")[-1].split(".")[0].split("-")[-1]
    save_dir_list = args.input[0].split("/")[:-3]
    save_dir = '/'.join(save_dir_list)
    save_name = save_dir + '/' + checkpoint_num + '_' + trial_num + '.pdf'

    if args.plot_flight:

        if args.remember:
            plotter.pickle_flights(args.remember)
        print ("Plot flight")
        f, _ = plotter.plot_flights2()
        f.set_size_inches(8,6)
        #f.set_dpi(300)
        if args.save_fig:
            plt.savefig(save_name)
        else:
            plt.show()

        #plt.figure()
        #plotter.plot_rate_spectrum()
    elif args.plot_progress:
        print ("Plot Progress")
        plotter.plot_progress(args.remember)


    if args.metrics:
        plotter.print_metrics()

