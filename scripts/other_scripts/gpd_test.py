#!/usr/bin/env python3

# Author: Jack Schultz

# To run this node in a given namespace with rosrun (for example 'my_gen3'), start a Kortex driver and then run : 
# rosrun kortex_examples example_moveit_trajectories.py __ns:=my_gen3

import sys
import pickle
import rospy
import moveit_commander
import moveit_msgs.msg
from math import pi
import tf2_ros, tf
import roslaunch
from gpd_ros.msg import GraspConfigList, GraspConfig
from geometry_msgs.msg import Pose, PoseStamped
import numpy as np
import tf2_geometry_msgs
from math import pi

class gpd_test():
    """gpd_test"""
    def __init__(self):
        # Some Initial Parameters
        self.grasp = None
        self.another_grasp = None

        # Initialize the node
        super(gpd_test, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('gpd_test')

        # Subscribers
        self.tfBuffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tfBuffer)
        rospy.Subscriber("/detect_grasps/clustered_grasps", GraspConfigList, self.callback)
        # rospy.Subscriber("/detect_grasps/plot_grasps", GraspConfigList, self.another_callback)

        # Publishers

        # Initialize the robot and MoveIt. Check for errors.
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
            print (e)
            self.is_init_success = False
        else:
            self.is_init_success = True

    # Subscriber Callbacks
    def callback(self, msg):
        self.grasp = msg

    def another_callback(self, msg):
        self.another_grasp = msg

    def reach_named_position(self, target):
        arm_group = self.arm_group
        
        # Going to one of those targets
        rospy.loginfo("Going to named target " + target)
        # Set the target
        arm_group.set_named_target(target)
        # Plan the trajectory
        (success_flag, trajectory_message, planning_time, error_code) = arm_group.plan()
        # Execute the trajectory and block while it's not finished
        arm_group.execute(trajectory_message, wait=True)

    def reach_named_joint_position(self, target, tolerance):
        arm_group = self.arm_group

        path = '/home/labuser/ros_ws/src/raf/set_positions/' + target + '.pkl'
        try:
            with open(path, 'rb') as handle:
                joint_positions = pickle.load(handle)
            handle.close()
        except:
            rospy.logerr("Named joint position does not exist.")

        # Set the goal joint tolerance
        self.arm_group.set_goal_joint_tolerance(tolerance)

        arm_group.set_joint_value_target(joint_positions)
        
        # Plan and execute in one command
        arm_group.go(wait=True)

    def reach_joint_angles(self, tolerance):
        arm_group = self.arm_group

        # Get the current joint positions
        joint_positions = arm_group.get_current_joint_values()
        rospy.loginfo("Printing current joint positions before movement :")
        for p in joint_positions: rospy.loginfo(p)

        # Set the goal joint tolerance
        self.arm_group.set_goal_joint_tolerance(tolerance)

        # Set the joint target configuration
        if self.degrees_of_freedom == 6:
            joint_positions[0] = -1.555139602746321
            joint_positions[1] = 0.5634708244906381
            joint_positions[2] = -1.513827582652576
            joint_positions[3] = 0.003350569000584251
            joint_positions[4] = -0.3392606238974132
            joint_positions[5] = 1.5689993588493656
        arm_group.set_joint_value_target(joint_positions)
        
        # Plan and execute in one command
        arm_group.go(wait=True)

        # Show joint positions after movement
        new_joint_positions = arm_group.get_current_joint_values()
        rospy.loginfo("Printing current joint positions after movement :")
        for p in new_joint_positions: rospy.loginfo(p)

    def remember_joint_angles(self, target):
        arm_group = self.arm_group

        joint_positions = arm_group.get_current_joint_values()

        # Remember the current joint positions
        path = '/home/labuser/ros_ws/src/raf/set_positions/' + target + '.pkl'
        f = open(path,"wb")
        pickle.dump(joint_positions, f)
        f.close()

        # Remember the current joint positions (Only works in the current session)
        # arm_group.remember_joint_values(target)

        rospy.loginfo("Joint angles remembered as named position: " + target)

    def reach_gripper_position(self, relative_position):
        gripper_group = self.gripper_group
        
        # We only have to move this joint because all others are mimic!
        gripper_joint = self.robot.get_joint(self.gripper_joint_name)
        gripper_max_absolute_pos = gripper_joint.max_bound()
        gripper_min_absolute_pos = gripper_joint.min_bound()
        gripper_joint.move(relative_position * (gripper_max_absolute_pos - gripper_min_absolute_pos) + gripper_min_absolute_pos, True)

    def get_tf(self, parent, child):
        try:
            trans = self.tfBuffer.lookup_transform(parent, child, rospy.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            trans = None
        return trans

    def transformPose(self, transform):

        # TODO: Be able to handle multiple grasps
        # Convert test grasp to a Pose
        # Position
        pose = Pose()
        pose.position.x = self.grasp.grasps[0].position.x
        pose.position.y = self.grasp.grasps[0].position.y
        pose.position.z = self.grasp.grasps[0].position.z

        # Orientation test
        # apX = 0.8725858509103502
        # apY = -0.22415247774347494
        # apZ = -0.433992626103855
        # bX = 0.1800014348402813
        # bY = 0.9735224173109384
        # bZ = -0.140902755290685
        # aX = 0.45408525217903695
        # aY = 0.04483045521213959
        # aZ = 0.8898296545063963

        apX = self.grasp.grasps[0].approach.x
        apY = self.grasp.grasps[0].approach.y
        apZ = self.grasp.grasps[0].approach.z
        bX = self.grasp.grasps[0].binormal.x
        bY = self.grasp.grasps[0].binormal.y
        bZ = self.grasp.grasps[0].binormal.z
        aX = self.grasp.grasps[0].axis.x
        aY = self.grasp.grasps[0].axis.y
        aZ = self.grasp.grasps[0].axis.z

        print("HERE!!!!!")

        R1 = np.array([[bX, bY, bZ, pose.position.x],[aX, aY, aZ, pose.position.y], [apX, apY, apZ, pose.position.z], [0, 0, 0, 1]])
        R2 = np.array([[-1, 0, 0, 0],[0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        R = np.matmul(R2, R1)
        print(R.shape)
        print(R)

        # Rx = tf.transformations.rotation_matrix(0, xaxis, [pose.position.x, pose.position.y, pose.position.z])
        # print(Rx)
        # Ry = tf.transformations.rotation_matrix(0, yaxis, [pose.position.x, pose.position.y, pose.position.z])
        # print(Ry)
        # Rz = tf.transformations.rotation_matrix(0, zaxis, [pose.position.x, pose.position.y, pose.position.z])
        # print(Rz)
        # R = tf.transformations.concatenate_matrices(Rx, Ry, Rz)
        # print(R)
        q = tf.transformations.quaternion_from_matrix(R)
        print(q)
        print("HERE!!!!!")

        # Orientation
        # xaxis = (self.grasp.grasps[0].approach.x, self.grasp.grasps[0].approach.y, self.grasp.grasps[0].approach.z)
        # yaxis = (self.grasp.grasps[0].binormal.x, self.grasp.grasps[0].binormal.y, self.grasp.grasps[0].binormal.z)
        # zaxis = (self.grasp.grasps[0].axis.x, self.grasp.grasps[0].axis.y, self.grasp.grasps[0].axis.z)
        # Rx = tf.transformations.rotation_matrix(0, xaxis)
        # Ry = tf.transformations.rotation_matrix(0, yaxis)
        # Rz = tf.transformations.rotation_matrix(0, zaxis)
        # R = tf.transformations.concatenate_matrices(Rx, Ry, Rz)
        # q = tf.transformations.quaternion_from_matrix(R)

        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]

        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "base_link"
        pose_stamped.pose = pose

        # Test stuff
        # arm_group = self.arm_group
        # current_pose = arm_group.get_current_pose()
        # pose_stamped.pose.orientation = current_pose.pose.orientation

        # Transform to base_link
        pose_transformed = tf2_geometry_msgs.do_transform_pose(pose_stamped, transform)

        return pose_stamped, pose_transformed

    def test_transformPose(self, transform):
        # Test grasp
        grasp = GraspConfig()
        grasp.position.x = -0.028277372423956434
        grasp.position.y = 0.0017110335228118
        grasp.position.z = 0.612631365562229
        grasp.approach.x = -0.2246895241792321
        grasp.approach.y = 0.14959830991554654
        grasp.approach.z = 0.8835657043976959
        grasp.binormal.x = 0.38456854210254926
        grasp.binormal.y = 0.8388704528656468
        grasp.binormal.z = -0.04423562283249827
        grasp.axis.x = -0.8094288385908327
        grasp.axis.y = 0.35702954999925607
        grasp.axis.z = -0.2662860234373907

        # grasp.position.x = -0.025921236381638207
        # grasp.position.y = 0.0016001277032683197
        # grasp.position.z = 0.6203329274569048
        # grasp.approach.x = 0.20308605549077266
        # grasp.approach.y = 0.4010873632987023
        # grasp.approach.z = 0.8932440769841657
        # grasp.binormal.x = -0.9791021169652265
        # grasp.binormal.y = 0.09317916669056
        # grasp.binormal.z = 0.1807669423569078
        # grasp.axis.x = -0.010728402463162347
        # grasp.axis.y = -0.9112884120282381
        # grasp.axis.z = 0.411628875910863

        arm_group = self.arm_group
        temp = arm_group.get_current_pose()

        # Convert test grasp to a Pose
        # Position
        pose = Pose()
        pose.position.x = grasp.position.x
        pose.position.y = grasp.position.y
        pose.position.z = grasp.position.z

        # Orientation
        xaxis = (grasp.approach.x, grasp.approach.y, grasp.approach.z)
        yaxis = (grasp.binormal.x, grasp.binormal.y, grasp.binormal.z)
        zaxis = (grasp.axis.x, grasp.axis.y, grasp.axis.z)
        Rx = tf.transformations.rotation_matrix(0, xaxis)
        Ry = tf.transformations.rotation_matrix(0, yaxis)
        Rz = tf.transformations.rotation_matrix(0, zaxis)
        R = tf.transformations.concatenate_matrices(Rx, Ry, Rz)
        q = tf.transformations.quaternion_from_matrix(R)

        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]

        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "camera_color_frame"
        pose_stamped.pose = pose

        # Transform to base_link
        pose_transformed = tf2_geometry_msgs.do_transform_pose(pose_stamped, transform)

        return pose_stamped, pose_transformed

    def get_cartesian_pose(self):
        arm_group = self.arm_group

        # Get the current pose and display it
        pose = arm_group.get_current_pose()
        rospy.loginfo("Actual cartesian pose is : ")
        rospy.loginfo(pose.pose)

        return pose.pose

    def reach_cartesian_pose(self, pose, tolerance, constraints):
        arm_group = self.arm_group
        
        # Set the tolerance
        arm_group.set_goal_position_tolerance(tolerance)

        # Set the trajectory constraint if one is specified
        if constraints is not None:
            arm_group.set_path_constraints(constraints)

        # Get the current Cartesian Position
        arm_group.set_pose_target(pose)

        # Plan and execute
        rospy.loginfo("Planning and going to the Cartesian Pose")
        arm_group.go(wait=True)

    def spinOnce(self):
        r = rospy.Rate(10)
        r.sleep()

def main():
    #################### Pseudo Code ####################
    # 1) Open gripper and move to view pose (joint state)
    # 2) Run GPD - get hand pose
    # 3) Listen for tf between camera frame and robot base frame
    # 4) Transform hand pose into robot base frame
    # 5) Send robot to hand pose
    # 6) Close gripper
    # 7) Move to release pose (joint state)
    # 8) Open gripper
    #####################################################
    run = gpd_test()

    # 1) Open gripper and move to view pose (joint state)
    rospy.loginfo("Opening the gripper...")
    run.reach_gripper_position(0)

    position = 'gpd-view-overhead'
    rospy.loginfo("Reaching Named Joint Position: (" + position + ")...")
    run.reach_named_joint_position(position, tolerance=0.01)

    # position = 'gpd-view'
    # rospy.loginfo("Reaching Named Joint Position: (" + position + ")...")
    # run.reach_named_joint_position(position, tolerance=0.01)

    # 2) TODO: Run GPD - get hand pose (Goal is to get one grasp. Might need to modify GPD source code)
    # uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    # roslaunch.configure_logging(uuid)
    # launch = roslaunch.parent.ROSLaunchParent(uuid, ["/home/labuser/ros_ws/src/gpd_ros/launch/kinova.launch"])
    # launch.start()
    print("Waiting for grasp...")
    while run.grasp is None:
        run.spinOnce()
    # launch.shutdown()

    print(run.grasp)

    # 3) Listen for tf between camera frame and robot base frame
    parent = 'base_link'
    child = 'base_link'
    # child = 'pcl_frame'
    # child = 'camera_color_frame'
    rospy.loginfo("Getting transformation from " + parent +  "to " + child + "frame...")
    trans = run.get_tf(parent, child)

    # 4) TODO: Transform hand pose into robot base frame
    # grasp, grasp_pose = run.test_transformPose(trans)
    grasp, grasp_pose = run.transformPose(trans)
    print("\n----------------------------")
    print("Grasp: ")
    print(grasp)
    print("\nPose: ")
    print(grasp_pose)
    print("------------------------------\n")

    # 5) Send robot to hand pose
    # TODO: Compute inverse kinematics for multiple grasps
    # TODO: Filter out impossible grasps
    # TODO: Choose grasp that is closest to current position (or something else)
    rospy.loginfo("Reaching Grasp Pose...")
    run.reach_cartesian_pose(pose=grasp_pose, tolerance=0.01, constraints=None)

    # 6) Close gripper
    rospy.loginfo("Closing the gripper 60%...")
    run.reach_gripper_position(0.65)

    # 7) Move to release pose (joint state)
    # position = 'gpd-view-overhead'
    # rospy.loginfo("Reaching Named Joint Position: (" + position + ")...")
    # run.reach_named_joint_position(position, tolerance=0.01)

    # position = 'gpd-view'
    position = 'gpd-view-overhead'
    rospy.loginfo("Reaching Named Joint Position: (" + position + ")...")
    run.reach_named_joint_position(position, tolerance=0.01)

    # position = 'release'
    # rospy.loginfo("Reaching Named Joint Position: (" + position + ")...")
    # run.reach_named_joint_position(position, tolerance=0.01)

    # 8) Open gripper
    # rospy.loginfo("Opening the gripper...")
    # run.reach_gripper_position(0)

    ####################################################################################################

    # if success:
    #     rospy.loginfo("Reaching Cartesian Pose...")
        
    #     actual_pose = example.get_cartesian_pose()
    #     actual_pose.position.x = -0.013764162486994358
    #     actual_pose.position.y = -0.015074951100195615
    #     actual_pose.position.z = 0.6141214859825755
    #     success &= example.reach_cartesian_pose(pose=actual_pose, tolerance=0.01, constraints=None)
    #     print (success)
        
    # if example.degrees_of_freedom == 7 and success:
    #     rospy.loginfo("Reach Cartesian Pose with constraints...")
    #     # Get actual pose
    #     actual_pose = example.get_cartesian_pose()
    #     actual_pose.position.y -= 0.3
        
    #     # Orientation constraint (we want the end effector to stay the same orientation)
    #     constraints = moveit_msgs.msg.Constraints()
    #     orientation_constraint = moveit_msgs.msg.OrientationConstraint()
    #     orientation_constraint.orientation = actual_pose.orientation
    #     constraints.orientation_constraints.append(orientation_constraint)

    #     # Send the goal
    #     success &= example.reach_cartesian_pose(pose=actual_pose, tolerance=0.01, constraints=constraints)

if __name__ == '__main__':
    main()
