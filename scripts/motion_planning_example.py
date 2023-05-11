#!/usr/bin/env python3

# This script is an example motion planning script for controlling the Kinova Gen3 using Moveit.

# Author: Jack Schultz
# Created 3/10/2023

import rospy, sys
import numpy as np


import moveit_commander
import moveit_msgs.msg
import tf2_ros
from tf.transformations import quaternion_slerp
from math import sqrt, inf, degrees, radians
import copy

class executeAction(object):
    def __init__(self):
        # Params

        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(15)

        # Publishers

        # Subscribers

        self.tfBuffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tfBuffer)

        # Robot Stuff
        moveit_commander.roscpp_initialize(sys.argv)
        try:
            self.is_gripper_present = rospy.get_param(rospy.get_namespace() + "is_gripper_present", False)
            if self.is_gripper_present:
                gripper_joint_names = rospy.get_param(rospy.get_namespace() + "gripper_joint_names", [])
                self.gripper_joint_name = gripper_joint_names[0]
            else:
                self.gripper_joint_name = ""
            self.degrees_of_freedom = rospy.get_param(rospy.get_namespace() + "degrees_of_freedom", 7)

            # Create the MoveItInterface necessary objects
            arm_group_name = "arm"
            self.robot = moveit_commander.RobotCommander("robot_description")
            self.scene = moveit_commander.PlanningSceneInterface(ns=rospy.get_namespace())
            self.arm_group = moveit_commander.MoveGroupCommander(arm_group_name, ns=rospy.get_namespace())
            self.display_trajectory_publisher = rospy.Publisher(rospy.get_namespace() + 'move_group/display_planned_path',
                                                            moveit_msgs.msg.DisplayTrajectory,
                                                            queue_size=20)

            if self.is_gripper_present:
                gripper_group_name = "gripper"
                self.gripper_group = moveit_commander.MoveGroupCommander(gripper_group_name, ns=rospy.get_namespace())

            rospy.loginfo("Initializing node in namespace " + rospy.get_namespace())
        except Exception as e:
            print(e)
            self.success = False
        else:
            self.success = True

    def init_set_positions(self):
        self.home = [0, 0.2619305484828358, -2.2690151129602745, 0, 0.9598884114404127, 1.5707802146703016]
        self.pose1 = [-0.6636096762192523, 0.1630385221975692, -1.7422927157194215, -1.1726596027824483, 0.7453946209262864, 2.6293639779494784]
        self.pose2 = [1.0856278978328227, 0.32326361444055146, -1.5283498000767777, 1.436878206115826, 1.0926278836000414, 1.155850928404359]

    ### ROBOT METHODS ###

    def compute_waypoints(self, P, n=5):
        '''This function generates n waypoints between the current pose and a specified pose
        using linear interpolarion for position and SLERP for quaternion interpolation.'''

        # Get current robot pose
        wpose = self.get_cartesian_pose()

        # Position
        x = np.linspace(wpose.position.x, P.position.x, n)
        y = np.linspace(wpose.position.y, P.position.y, n)
        z = np.linspace(wpose.position.z, P.position.z, n)

        # Orientation
        q_lin = np.linspace(0.0, 1.0, n)
        q0 = [wpose.orientation.x, wpose.orientation.y, wpose.orientation.z, wpose.orientation.w]
        q1 = [P.orientation.x, P.orientation.y, P.orientation.z, P.orientation.w] 

        waypoints = []
        for i in range(n):
            waypoint = copy.deepcopy(wpose)
            waypoint.position.x = x[i]
            waypoint.position.y = y[i]
            waypoint.position.z = z[i]
            [waypoint.orientation.x, waypoint.orientation.y, waypoint.orientation.z, waypoint.orientation.w] = quaternion_slerp(q0, q1, q_lin[i])
            waypoints.append(waypoint)

        return waypoints

    def move(self, goal_type, goal, tolerance=0.01, vel=0.5, accel=0.5, attempts=10, time=5.0, constraints=None):
        arm_group = self.arm_group
            

        # Set parameters
        self.arm_group.set_max_velocity_scaling_factor(vel)
        self.arm_group.set_max_acceleration_scaling_factor(accel)
        self.arm_group.set_num_planning_attempts(attempts)
        self.arm_group.set_planning_time(time)
        
        if goal_type == 'pose':
            arm_group.clear_pose_targets()
            # Set the tolerance
            arm_group.set_goal_position_tolerance(tolerance)

            # Set the trajectory constraint if one is specified
            if constraints is not None:
                arm_group.set_path_constraints(constraints)

            # Get the current Cartesian Position
            arm_group.set_pose_target(goal)

            # Plan & Execute
            (success, plan, planning_time, error_code) = arm_group.plan()
            if success:
                print("Planning Successful.")
                print(f"Planning time: {planning_time}")
                print("Executing Plan...")
                success = arm_group.execute(plan, wait=True)
                arm_group.stop()
                arm_group.clear_pose_targets()
            
        elif goal_type == 'joint':
            # Get the current joint positions
            joint_positions = arm_group.get_current_joint_values()

            # Set the goal joint tolerance
            self.arm_group.set_goal_joint_tolerance(tolerance)

            # Set the joint target configuration
            joint_positions[0] = goal[0]
            joint_positions[1] = goal[1]
            joint_positions[2] = goal[2]
            joint_positions[3] = goal[3]
            joint_positions[4] = goal[4]
            joint_positions[5] = goal[5]
            arm_group.set_joint_value_target(joint_positions)

            # Plan & Execute
            (success, plan, planning_time, error_code) = arm_group.plan()
            if success:
                print("Planning Successful.")
                print(f"Planning time: {planning_time}")
                print("Executing Plan...")
                success = arm_group.execute(plan, wait=True)
                arm_group.stop()

        elif goal_type == 'path':
            # Clear old pose targets
            arm_group.clear_pose_targets()

            # Clear max cartesian speed
            arm_group.clear_max_cartesian_link_speed()

            # Set the tolerance
            arm_group.set_goal_position_tolerance(tolerance)

            # Set the trajectory constraint if one is specified
            if constraints is not None:
                arm_group.set_path_constraints(constraints)

            eef_step = 0.01
            jump_threshold = 0.0
            (plan, fraction) = arm_group.compute_cartesian_path(goal, eef_step, jump_threshold)
            success = arm_group.execute(plan, wait=True)
            arm_group.stop()

        elif goal_type == 'gripper':
            # We only have to move this joint because all others are mimic!
            gripper_joint = self.robot.get_joint(self.gripper_joint_name)
            gripper_max_absolute_pos = gripper_joint.max_bound()
            gripper_min_absolute_pos = gripper_joint.min_bound()
            success = gripper_joint.move(goal * (gripper_max_absolute_pos - gripper_min_absolute_pos) + gripper_min_absolute_pos, True)
        
        else:
            rospy.ERROR("Invalid Goal Type.")

        return success
    
    def get_cartesian_pose(self):
        arm_group = self.arm_group

        # Get the current pose and display it
        pose = arm_group.get_current_pose()

        return pose.pose

    def get_tf(self, parent, child):
        try:
            trans = self.tfBuffer.lookup_transform(parent, child, rospy.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            trans = None
        return trans

    def spinOnce(self):
        r = rospy.Rate(10)
        r.sleep()

def main():
    """ RAF Execute Action """
    rospy.init_node("raf_execute_action", anonymous=True)
    run = executeAction()
    run.init_set_positions()

    success = True

    # Move to home via joints and get pose
    if success:
        success &= run.move('joint', run.home, tolerance=0.01, vel=1.0, accel=1.0, attempts=10, time=5.0, constraints=None)
        home = run.get_cartesian_pose()

    # Move to pose 1 via joints and get pose
    if success:
        success &= run.move('joint', run.pose1, tolerance=0.01, vel=1.0, accel=1.0, attempts=10, time=5.0, constraints=None)
        pose1 = run.get_cartesian_pose()

    # Move to pose 2 via joints and get pose
    if success:
        success &= run.move('joint', run.pose2, tolerance=0.01, vel=1.0, accel=1.0, attempts=10, time=5.0, constraints=None)
        pose2 = run.get_cartesian_pose()

    # Move back to home
    if success:
        success &= run.move('pose', home, tolerance=0.01, vel=1.0, accel=1.0, attempts=10, time=5.0, constraints=None)

    # Create waypoints between home and pose 1, then plan and execute trajectory
    if success:
        waypoints = run.compute_waypoints(pose1, n=5)
        success &= run.move('path', waypoints, tolerance=0.01, vel=1.0, accel=0.9, attempts=10, time=5.0, constraints=None)

    # Create waypoints between pose 1 and pose 2, then plan and execute trajectory
    if success:
        waypoints = run.compute_waypoints(pose2, n=5)
        success &= run.move('path', waypoints, tolerance=0.01, vel=1.0, accel=0.8, attempts=10, time=5.0, constraints=None)

    # Create waypoints between pose 2 and home, then plan and execute trajectory
    if success:
        waypoints = run.compute_waypoints(home, n=5)
        success &= run.move('path', waypoints, tolerance=0.01, vel=1.0, accel=0.9, attempts=10, time=5.0, constraints=None)

if __name__ == '__main__':
    sys.exit(main())