#!/usr/bin/python
#\file          follow_q_traj1.py
#\brief      Follow a joint angle trajectory.
#                     WARNING: This code does not work as of Nov 26 2019
#                     since PreComputedJointTrajectory (used in implementing follow_joint_trajectory)
#                     rejects trajectories that do not
#                     have 1msec timesteps intervals between all trajectory points.
#                     cf.
#                     https://github.com/Kinovarobotics/matlab_kortex/blob/master/simplified_api/documentation/precomputed_joint_trajectories.md#hard-limits-and-conditions-to-respect
#\author     Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date          Nov.25, 2019

import argparse
import rospy
import actionlib
import control_msgs.msg
import trajectory_msgs.msg
import sensor_msgs.msg
import sys, copy
import numpy as np

def add_point(goal, time, positions, velocities):
    point= trajectory_msgs.msg.JointTrajectoryPoint()
    point.positions= copy.deepcopy(positions)
    point.velocities= copy.deepcopy(velocities)
    point.time_from_start= rospy.Duration(time)
    goal.trajectory.points.append(point)

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
        '-f', '--file', metavar='PATH', required=True,
        help='path to input file'
    )
    parser.add_argument(
        '-l', '--loops', type=int, default=1,
        help='number of times to loop the input file. 0=infinite.'
    )
    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node('gen3_test')
    joint_names= ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']

    # Load data from file
    data = np.loadtxt(args.file, skiprows=1, dtype='float', delimiter=",")

    time = data[0:500,0]
    joints = [data[0:500,1], data[0:500,2], data[0:500,3], data[0:500,4], data[0:500,5], data[0:500,6]]

    # Initialize
    client= actionlib.SimpleActionClient('/my_gen3/gen3_joint_trajectory_controller/follow_joint_trajectory', control_msgs.msg.FollowJointTrajectoryAction)
    client.cancel_goal()     #Ensure to cancel the ongoing goal.

    # Wait some seconds for the head action server to start or exit
    if not client.wait_for_server(rospy.Duration(5.0)):
        rospy.logerr('Exiting - Joint Trajectory Action Server Not Found')
        rospy.signal_shutdown('Action Server not found')
        sys.exit(1)

    goal= control_msgs.msg.FollowJointTrajectoryGoal()
    #goal.goal_time_tolerance= rospy.Time(0.1)
    goal.trajectory.joint_names= joint_names

    angles = rospy.wait_for_message('/my_gen3/joint_states', sensor_msgs.msg.JointState, 5.0).position
    for i in range(len(time)):
        add_point(goal, time[i], [joints[0][i], joints[1][i], joints[2][i], joints[3][i], joints[4][i], joints[5][i]], [0.0]*6)

    print(len(goal.trajectory.points))

    goal.trajectory.header.stamp= rospy.Time.now()
    client.send_goal(goal)
    #client.cancel_goal()
    #client.wait_for_result(timeout=rospy.Duration(20.0))

    print(client.get_result())

    #rospy.signal_shutdown('Done.')

if __name__ == '__main__':
    main()