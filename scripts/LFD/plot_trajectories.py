import matplotlib.pyplot as plt
import argparse
import rospy
import numpy as np
from pathlib import Path

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
  record_trajectories.py; plot_trajectories_multi.py.
    """
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt,
                                     description=main.__doc__,
                                     epilog=epilog)
    parser.add_argument(
        '-f', '--file', metavar='PATH', required=True,
        help='path to input file'
    )
    parser.add_argument(
        '-l', '--loops', type=int, default=1,
        help='number of times to loop the input file. 0=infinite.'
    )
    args = parser.parse_args(rospy.myargv()[1:])

    filename = Path(args.file).with_suffix('')

    data = np.loadtxt(args.file, skiprows=1, dtype='float', delimiter=",")

    time = data[:,0]
    joint1 = data[:,1]
    joint2 = data[:,2]
    joint3 = data[:,3]
    joint4 = data[:,4]
    joint5 = data[:,5]
    joint6 = data[:,6]
    gripper = data[:,7]

    # Separate plots for each joint in same figure [1,7]
    fig, axs = plt.subplots(1, 7)
    fig.set_size_inches(25, 5)
    axs[0].plot(time, joint1, c = 'b')
    axs[0].set_title('Joint 1')
    axs[1].plot(time, joint2, c = 'g')
    axs[1].set_title('Joint 2')
    axs[2].plot(time, joint3, c = 'r')
    axs[2].set_title('Joint 3')
    axs[3].plot(time, joint4, c = 'c')
    axs[3].set_title('Joint 4')
    axs[4].plot(time, joint5, c = 'm')
    axs[4].set_title('Joint 5')
    axs[5].plot(time, joint6, c = 'y')
    axs[5].set_title('Joint 6')
    axs[6].plot(time, gripper, c = 'k')
    axs[6].set_title('Gripper')

    for ax in axs.flat:
        ax.set(xlabel='Time (s)', ylabel='Joint Angle (rad)')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    fig_file = str(filename) + '_fig.png'
    plt.savefig(fig_file)
    plt.show()

    # Separate plots for each joint in same figure
    # fig, axs = plt.subplots(2, 3)
    # axs[0, 0].plot(time, joint1)
    # axs[0, 0].set_title('Joint 1')
    # axs[0, 1].plot(time, joint2, 'tab:orange')
    # axs[0, 1].set_title('Joint 2')
    # axs[0, 2].plot(time, joint3, 'tab:green')
    # axs[0, 2].set_title('Joint 3')
    # axs[1, 0].plot(time, joint4, 'tab:red')
    # axs[1, 0].set_title('Joint 4')
    # axs[1, 1].plot(time, joint5, 'tab:green')
    # axs[1, 1].set_title('Joint 5')
    # axs[1, 2].plot(time, joint6, 'tab:blue')
    # axs[1, 2].set_title('Joint 6')

    # for ax in axs.flat:
    #     ax.set(xlabel='Time (s)', ylabel='Joint Angle (rad)')

    # for ax in axs.flat:
    #     ax.label_outer()

    # fig_file = str(filename) + '_fig.png'
    # plt.savefig(fig_file)
    # plt.show()

    # All Joints on One Plot
    # plt.title('All Joints')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Joint Angle (rad)')
    # plt.plot(time, joint1, c = 'b')
    # plt.plot(time, joint2, c = 'g')
    # plt.plot(time, joint3, c = 'r')
    # plt.plot(time, joint4, c = 'c')
    # plt.plot(time, joint5, c = 'm')
    # plt.plot(time, joint6, c = 'y')
    # plt.plot(time, gripper, c = 'k')

    # plt.show()

if __name__ == '__main__':
    main()