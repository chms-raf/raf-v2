import matplotlib.pyplot as plt
import argparse
import rospy
import numpy as np
import os

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

def main():
    """RSDK Joint Position Example: File Playback
    Uses Joint Position Control mode to play back a series of
    recorded joint and gripper positions.
    Run the joint_recorder.py example first to create a recording
    file for use with this example. This example uses position
    control to replay the recorded positions in sequence.
    Note: This version of the playback example simply drives the
    joints towards the next position at each time stamp. Because
    it uses Position Control it will not attempt to adjust the
    movement speed to hit set points "on time".
    """
    epilog = """
Related examples:
  record_trajectories.py; plot_trajectories.py.
    """
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt,
                                     description=main.__doc__,
                                     epilog=epilog)
    parser.add_argument(
        '-d', '--directory', metavar='PATH', required=True,
        help='path to input file'
    )
    parser.add_argument(
        '-l', '--loops', type=int, default=1,
        help='number of times to loop the input file. 0=infinite.'
    )
    args = parser.parse_args(rospy.myargv()[1:])
    ext = ('.txt')

    plot_means = False

    # Separate plots for each joint in same figure
    fig, axs = plt.subplots(4, 3)
    fig.set_size_inches(15, 10)

    thresh = .0001
    start_margin = 0
    end_margin = 15

    keys = list()
    joint1 = list()
    joint2 = list()
    joint3 = list()
    joint4 = list()
    joint5 = list()
    joint6 = list()
    X = list()
    Y = list()
    Z = list()
    for file in os.listdir(args.directory):
        if file.endswith(ext):
            data = np.loadtxt(str(args.directory) + '/' + file, skiprows=1, dtype='float', delimiter=",")

            time = data[:,0]
            joints = [data[:,1], data[:,2], data[:,3], data[:,4], data[:,5], data[:,6]]
            effector = [data[:,7], data[:,8], data[:,9]]
            gripper = data[:,10]

            start_time_idx = np.inf
            for joint in joints:
                diff_joint = np.diff(joint)
                for i in range(len(diff_joint)):
                    if abs(diff_joint[i]) > thresh and i < start_time_idx:
                        start_time_idx = i-start_margin

            # Find end time
            diff_gripper = np.diff(gripper)
            for i in reversed(range(len(diff_gripper))):
                if abs(diff_gripper[i]) > thresh:
                    end_time_idx = i+end_margin

            # Trim trajectories
            time = time[start_time_idx:end_time_idx]
            for i in range(len(joints)):
                joints[i] = joints[i][start_time_idx:end_time_idx]
            for i in range(len(effector)):
                effector[i] = effector[i][start_time_idx:end_time_idx]
            gripper = gripper[start_time_idx:end_time_idx]

            # Normalize Time
            time = (time-min(time))/(max(time)-min(time))
  
            # Plot each trajectory
            axs[0,0].plot(time, joints[0], c = 'tab:blue')
            axs[0,1].plot(time, joints[1], c = 'tab:orange')
            axs[0,2].plot(time, joints[2], c = 'tab:green')
            axs[1,0].plot(time, joints[3], c = 'tab:olive')
            axs[1,1].plot(time, joints[4], c = 'tab:purple')
            axs[1,2].plot(time, joints[5], c = 'tab:brown')
            axs[2,0].plot(time, effector[0], c = 'r')
            axs[2,1].plot(time, effector[1], c = 'g')
            axs[2,2].plot(time, effector[2], c = 'b')
            axs[3,0].plot(time, gripper, c = 'k')

            # Save data for dictionary
            keys.append(os.path.splitext(file)[0])
            joint1.append(np.array(joints[0]))
            joint2.append(np.array(joints[1]))
            joint3.append(np.array(joints[2]))
            joint4.append(np.array(joints[3]))
            joint5.append(np.array(joints[4]))
            joint6.append(np.array(joints[5]))
            X.append(np.array(effector[0]))
            Y.append(np.array(effector[1]))
            Z.append(np.array(effector[2]))
    
    # Create dictionaries
    num_files = len(keys)

    joint1_dict = {}
    joint2_dict = {}
    joint3_dict = {}
    joint4_dict = {}
    joint5_dict = {}
    joint6_dict = {}
    X_dict = {}
    Y_dict = {}
    Z_dict = {}
    for i in range(len(keys)):
        joint1_dict[i] = joint1[i]
        joint2_dict[i] = joint2[i]
        joint3_dict[i] = joint3[i]
        joint4_dict[i] = joint4[i]
        joint5_dict[i] = joint5[i]
        joint6_dict[i] = joint6[i]
        X_dict[i] = X[i]
        Y_dict[i] = Y[i]
        Z_dict[i] = Z[i]

    # Resample and interpolate
    # TODO: Get max length of arrays instead of hard coding. Will need to do for longer trajectories
    samples = 100
    joint1_interped = {k: np.interp(np.linspace(0, 1, samples), np.linspace(0, 1, len(joint1_dict[k])), data) for k, data in joint1_dict.items()}
    joint2_interped = {k: np.interp(np.linspace(0, 1, samples), np.linspace(0, 1, len(joint2_dict[k])), data) for k, data in joint2_dict.items()}
    joint3_interped = {k: np.interp(np.linspace(0, 1, samples), np.linspace(0, 1, len(joint3_dict[k])), data) for k, data in joint3_dict.items()}
    joint4_interped = {k: np.interp(np.linspace(0, 1, samples), np.linspace(0, 1, len(joint4_dict[k])), data) for k, data in joint4_dict.items()}
    joint5_interped = {k: np.interp(np.linspace(0, 1, samples), np.linspace(0, 1, len(joint5_dict[k])), data) for k, data in joint5_dict.items()}
    joint6_interped = {k: np.interp(np.linspace(0, 1, samples), np.linspace(0, 1, len(joint6_dict[k])), data) for k, data in joint6_dict.items()}
    X_interped = {k: np.interp(np.linspace(0, 1, samples), np.linspace(0, 1, len(X_dict[k])), data) for k, data in X_dict.items()}
    Y_interped = {k: np.interp(np.linspace(0, 1, samples), np.linspace(0, 1, len(Y_dict[k])), data) for k, data in Y_dict.items()}
    Z_interped = {k: np.interp(np.linspace(0, 1, samples), np.linspace(0, 1, len(Z_dict[k])), data) for k, data in Z_dict.items()}

    joint1_array = np.empty([num_files, samples])
    joint2_array = np.empty([num_files, samples])
    joint3_array = np.empty([num_files, samples])
    joint4_array = np.empty([num_files, samples])
    joint5_array = np.empty([num_files, samples])
    joint6_array = np.empty([num_files, samples])
    X_array = np.empty([num_files, samples])
    Y_array = np.empty([num_files, samples])
    Z_array = np.empty([num_files, samples])
    for i in range(num_files):
        joint1_array[i,:] = joint1_interped[i]
        joint2_array[i,:] = joint2_interped[i]
        joint3_array[i,:] = joint3_interped[i]
        joint4_array[i,:] = joint4_interped[i]
        joint5_array[i,:] = joint5_interped[i]
        joint6_array[i,:] = joint6_interped[i]
        X_array[i,:] = X_interped[i]
        Y_array[i,:] = Y_interped[i]
        Z_array[i,:] = Z_interped[i]
    
    # Compute mean
    joint1_mean = np.mean(joint1_array, axis=0)
    joint2_mean = np.mean(joint2_array, axis=0)
    joint3_mean = np.mean(joint3_array, axis=0)
    joint4_mean = np.mean(joint4_array, axis=0)
    joint5_mean = np.mean(joint5_array, axis=0)
    joint6_mean = np.mean(joint6_array, axis=0)
    X_mean = np.mean(X_array, axis=0)
    Y_mean = np.mean(Y_array, axis=0)
    Z_mean = np.mean(Z_array, axis=0)

    # Plot means
    time_lin = np.linspace(0, 1, samples)
    if plot_means:
        axs[0,0].plot(time_lin, joint1_mean, c = 'k', linewidth=2)
        axs[0,1].plot(time_lin, joint2_mean, c = 'k', linewidth=2)
        axs[0,2].plot(time_lin, joint3_mean, c = 'k', linewidth=2)
        axs[1,0].plot(time_lin, joint4_mean, c = 'k', linewidth=2)
        axs[1,1].plot(time_lin, joint5_mean, c = 'k', linewidth=2)
        axs[1,2].plot(time_lin, joint6_mean, c = 'k', linewidth=2)
        axs[2,0].plot(time_lin, X_mean, c = 'k', linewidth=2)
        axs[2,1].plot(time_lin, Y_mean, c = 'k', linewidth=2)
        axs[2,2].plot(time_lin, Z_mean, c = 'k', linewidth=2)

    
    # Resample mean trajectories so each point is increments of .001s
    end_time = 10 # TODO: This needs to be based on the actual trajectories somehow...
    time_traj = trunc(np.arange(0, end_time+.001, .001), decs=3)
    joint1_mean_interped = np.interp(time_traj, time_lin, joint1_mean)
    joint2_mean_interped = np.interp(time_traj, time_lin, joint2_mean)
    joint3_mean_interped = np.interp(time_traj, time_lin, joint3_mean)
    joint4_mean_interped = np.interp(time_traj, time_lin, joint4_mean)
    joint5_mean_interped = np.interp(time_traj, time_lin, joint5_mean)
    joint6_mean_interped = np.interp(time_traj, time_lin, joint6_mean)

    # Save mean trajectories to a file
    with open('traj_mean.txt', 'w') as f:
        f.write('time,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6\n')
        for i in range(len(time_traj)):
            f.write(str(time_traj[i]) + ',' + str(joint1_mean_interped[i]) + ',' + 
                    str(joint2_mean_interped[i]) + ',' + str(joint3_mean_interped[i]) + ',' + 
                    str(joint4_mean_interped[i]) + ',' + str(joint5_mean_interped[i]) + ',' + 
                    str(joint6_mean_interped[i]) + '\n')
    
    # Compute covariance
    joint1_cov_array = np.empty([num_files,samples])
    joint2_cov_array = np.empty([num_files,samples])
    joint3_cov_array = np.empty([num_files,samples])
    joint4_cov_array = np.empty([num_files,samples])
    joint5_cov_array = np.empty([num_files,samples])
    joint6_cov_array = np.empty([num_files,samples])
    X_cov_array = np.empty([num_files,samples])
    Y_cov_array = np.empty([num_files,samples])
    Z_cov_array = np.empty([num_files,samples])
    for i in range(num_files):
        for ii in range(samples):
            joint1_cov_array[i,ii] = (joint1_array[i,ii] - joint1_mean[ii])*np.transpose(joint1_array[i,ii] - joint1_mean[ii])
            joint2_cov_array[i,ii] = (joint2_array[i,ii] - joint2_mean[ii])*np.transpose(joint2_array[i,ii] - joint2_mean[ii])
            joint3_cov_array[i,ii] = (joint3_array[i,ii] - joint3_mean[ii])*np.transpose(joint3_array[i,ii] - joint3_mean[ii])
            joint4_cov_array[i,ii] = (joint4_array[i,ii] - joint4_mean[ii])*np.transpose(joint4_array[i,ii] - joint4_mean[ii])
            joint5_cov_array[i,ii] = (joint5_array[i,ii] - joint5_mean[ii])*np.transpose(joint5_array[i,ii] - joint5_mean[ii])
            joint6_cov_array[i,ii] = (joint6_array[i,ii] - joint6_mean[ii])*np.transpose(joint6_array[i,ii] - joint6_mean[ii])
            X_cov_array[i,ii] = (X_array[i,ii] - X_mean[ii])*np.transpose(X_array[i,ii] - X_mean[ii])
            Y_cov_array[i,ii] = (Y_array[i,ii] - Y_mean[ii])*np.transpose(Y_array[i,ii] - Y_mean[ii])
            Z_cov_array[i,ii] = (Z_array[i,ii] - Z_mean[ii])*np.transpose(Z_array[i,ii] - Z_mean[ii])
    joint1_cov = (np.sum(joint1_cov_array, axis=0)) / (num_files - 1)
    joint2_cov = (np.sum(joint2_cov_array, axis=0)) / (num_files - 1)
    joint3_cov = (np.sum(joint3_cov_array, axis=0)) / (num_files - 1)
    joint4_cov = (np.sum(joint4_cov_array, axis=0)) / (num_files - 1)
    joint5_cov = (np.sum(joint5_cov_array, axis=0)) / (num_files - 1)
    joint6_cov = (np.sum(joint6_cov_array, axis=0)) / (num_files - 1)
    X_cov = (np.sum(X_cov_array, axis=0)) / (num_files - 1)
    Y_cov = (np.sum(Y_cov_array, axis=0)) / (num_files - 1)
    Z_cov = (np.sum(Z_cov_array, axis=0)) / (num_files - 1)
    
    # Plot Covariance
    if plot_means:
        axs[0,0].plot(time_lin, joint1_mean + 2*np.sqrt(joint1_cov), c = 'k', linewidth=1)
        axs[0,0].plot(time_lin, joint1_mean - 2*np.sqrt(joint1_cov), c = 'k', linewidth=1)
        axs[0,1].plot(time_lin, joint2_mean + 2*np.sqrt(joint2_cov), c = 'k', linewidth=1)
        axs[0,1].plot(time_lin, joint2_mean - 2*np.sqrt(joint2_cov), c = 'k', linewidth=1)
        axs[0,2].plot(time_lin, joint3_mean + 2*np.sqrt(joint3_cov), c = 'k', linewidth=1)
        axs[0,2].plot(time_lin, joint3_mean - 2*np.sqrt(joint3_cov), c = 'k', linewidth=1)
        axs[1,0].plot(time_lin, joint4_mean + 2*np.sqrt(joint4_cov), c = 'k', linewidth=1)
        axs[1,0].plot(time_lin, joint4_mean - 2*np.sqrt(joint4_cov), c = 'k', linewidth=1)
        axs[1,1].plot(time_lin, joint5_mean + 2*np.sqrt(joint5_cov), c = 'k', linewidth=1)
        axs[1,1].plot(time_lin, joint5_mean - 2*np.sqrt(joint5_cov), c = 'k', linewidth=1)
        axs[1,2].plot(time_lin, joint6_mean + 2*np.sqrt(joint6_cov), c = 'k', linewidth=1)
        axs[1,2].plot(time_lin, joint6_mean - 2*np.sqrt(joint6_cov), c = 'k', linewidth=1)
        axs[2,0].plot(time_lin, X_mean + 2*np.sqrt(X_cov), c = 'k', linewidth=1)
        axs[2,0].plot(time_lin, X_mean - 2*np.sqrt(X_cov), c = 'k', linewidth=1)
        axs[2,1].plot(time_lin, Y_mean + 2*np.sqrt(Y_cov), c = 'k', linewidth=1)
        axs[2,1].plot(time_lin, Y_mean - 2*np.sqrt(Y_cov), c = 'k', linewidth=1)
        axs[2,2].plot(time_lin, Z_mean + 2*np.sqrt(Z_cov), c = 'k', linewidth=1)
        axs[2,2].plot(time_lin, Z_mean - 2*np.sqrt(Z_cov), c = 'k', linewidth=1)
    
    axs[0,0].set_title('Joint 1')
    axs[0,1].set_title('Joint 2')
    axs[0,2].set_title('Joint 3')
    axs[1,0].set_title('Joint 4')
    axs[1,1].set_title('Joint 5')
    axs[1,2].set_title('Joint 6')
    axs[2,0].set_title('X')
    axs[2,1].set_title('Y')
    axs[2,2].set_title('Z')
    axs[3,0].set_title('Gripper')

    for ax in axs.flat:
        ax.set(xlabel='Time (s)', ylabel='Joint Angle (rad)')

    axs[2,0].set(ylabel='Position (m)')
    axs[3,0].set(ylabel='Rel. Position')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    fig_file = str(args.directory) + '_fig.png'
    plt.savefig(str(args.directory) + '/' + fig_file)

    # Create second plot to replicate DGMP paper
    fig2, axs2 = plt.subplots(1, 2)
    fig2.set_size_inches(20, 7)

    time_lin = np.linspace(0, 1, samples)
    line1, = axs2[0].plot(time_lin, joint1_mean, c = 'tab:blue', linewidth=2, label='joint1')
    axs2[0].fill_between(time_lin, joint1_mean + 2*np.sqrt(joint1_cov), joint1_mean - 2*np.sqrt(joint1_cov), color='tab:blue', alpha=0.2)
    line2, = axs2[0].plot(time_lin, joint2_mean, c = 'tab:orange', linewidth=2, label='joint2')
    axs2[0].fill_between(time_lin, joint2_mean + 2*np.sqrt(joint2_cov), joint2_mean - 2*np.sqrt(joint2_cov), color='tab:orange', alpha=0.2)
    line3, = axs2[0].plot(time_lin, joint3_mean, c = 'tab:green', linewidth=2, label='joint3')
    axs2[0].fill_between(time_lin, joint3_mean + 2*np.sqrt(joint3_cov), joint3_mean - 2*np.sqrt(joint3_cov), color='tab:green', alpha=0.2)
    line4, = axs2[0].plot(time_lin, joint4_mean, c = 'tab:olive', linewidth=2, label='joint4')
    axs2[0].fill_between(time_lin, joint4_mean + 2*np.sqrt(joint4_cov), joint4_mean - 2*np.sqrt(joint4_cov), color='tab:olive', alpha=0.2)
    line5, = axs2[0].plot(time_lin, joint5_mean, c = 'tab:purple', linewidth=2, label='joint5')
    axs2[0].fill_between(time_lin, joint5_mean + 2*np.sqrt(joint5_cov), joint5_mean - 2*np.sqrt(joint5_cov), color='tab:purple', alpha=0.2)
    line6, = axs2[0].plot(time_lin, joint6_mean, c = 'tab:brown', linewidth=2, label='joint6')
    axs2[0].fill_between(time_lin, joint6_mean + 2*np.sqrt(joint6_cov), joint6_mean - 2*np.sqrt(joint6_cov), color='tab:brown', alpha=0.2)
    line7, = axs2[1].plot(time_lin, X_mean, c = 'r', linewidth=2, label='X')
    axs2[1].fill_between(time_lin, X_mean + 2*np.sqrt(X_cov), X_mean - 2*np.sqrt(X_cov), color='r', alpha=0.2)
    line8, = axs2[1].plot(time_lin, Y_mean, c = 'g', linewidth=2, label='Y')
    axs2[1].fill_between(time_lin, Y_mean + 2*np.sqrt(Y_cov), Y_mean - 2*np.sqrt(Y_cov), color='g', alpha=0.2)
    line9, = axs2[1].plot(time_lin, Z_mean, c = 'b', linewidth=2, label='Z')
    axs2[1].fill_between(time_lin, Z_mean + 2*np.sqrt(Z_cov), Z_mean - 2*np.sqrt(Z_cov), color='b', alpha=0.2)

    axs2[0].set_title('Joints')
    axs2[1].set_title('End Effector')
    axs2[0].set(ylabel='Joint Angle (rad)')
    axs2[1].set(ylabel='Positon (m)')
    for ax in axs2.flat:
        ax.set(xlabel='Time (s)')

    axs2[0].legend(handles=[line1, line2, line3, line4, line5, line6], loc='upper right')
    axs2[1].legend(handles=[line7, line8, line9], loc='upper right')

    fig_file2 = str(args.directory) + '_fig2.png'
    plt.savefig(str(args.directory) + '/' + fig_file2)
    plt.show()

if __name__ == '__main__':
    main()