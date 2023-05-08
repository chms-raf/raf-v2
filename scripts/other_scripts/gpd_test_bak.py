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

from sensor_msgs.msg import PointCloud2
import time
# from raf import point_cloud2_functions as pc
# from tools.point_cloud2_functions import pointcloud2_to_array
from pc_tools.point_cloud2_functions import pointcloud2_to_array, pointcloud2_to_xyz_array, array_to_pointcloud2
from random import randint
from gpd_ros import grasp_detection_node

class Robot():
    def __init__(self):
        # Initial parameters
        self.grasps = list()

        # Subscribers
        rospy.Subscriber("/detect_grasps/clustered_grasps", GraspConfigList, self.grasp_callback)
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
    def grasp_callback(self, msg):
        self.grasp = msg

    # Primary Methods
    def grasp_to_pose(self, tfBuffer):
        # Get transformation
        parent = 'base_link'
        child = 'base_link'
        # child = 'pcl_frame'
        # child = 'camera_color_frame'
        try:
            transform = tfBuffer.lookup_transform(parent, child, rospy.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("Error in transform lookup.")
            transform = None

        # Extra 180 degree rotation around gripper z axis for now
        R1 = np.array([[-1, 0, 0, 0],[0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        poses = []
        for grasp in self.grasps:
            pose = Pose()
            pose.position = grasp.position
            apX = grasp.approach.x
            apY = grasp.approach.y
            apZ = grasp.approach.z
            bX = grasp.binormal.x
            bY = grasp.binormal.y
            bZ = grasp.binormal.z
            aX = grasp.axis.x
            aY = grasp.axis.y
            aZ = grasp.axis.z
            R2 = np.array([[bX, bY, bZ, pose.position.x],[aX, aY, aZ, pose.position.y], [apX, apY, apZ, pose.position.z], [0, 0, 0, 1]])
            R = np.matmul(R1, R2)
            q = tf.transformations.quaternion_from_matrix(R)
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]

            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = "base_link"
            pose_stamped.pose = pose
            pose_transformed = tf2_geometry_msgs.do_transform_pose(pose_stamped, transform)
            poses.append(pose_transformed)

        return poses

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

    def reach_gripper_position(self, relative_position):
        gripper_joint = self.robot.get_joint(self.gripper_joint_name)
        gripper_max_absolute_pos = gripper_joint.max_bound()
        gripper_min_absolute_pos = gripper_joint.min_bound()
        gripper_joint.move(relative_position * (gripper_max_absolute_pos - gripper_min_absolute_pos) + gripper_min_absolute_pos, True)

class Cloud():
    def __init__(self):
        # Some Initial Parameters
        self.transformed_cloud = PointCloud2()

        # Subscribers
        self.tfBuffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tfBuffer)
        rospy.Subscriber("/camera/depth_registered/points", PointCloud2, self.callback)

        # Publishers
        self.cloud_pub = rospy.Publisher('/cloud_transformed', PointCloud2, queue_size=10)

    # Subscriber Callbacks
    def callback(self, msg):
        self.cloud = msg
        self.publish()

    # Primary Methods
    def transform_cloud(self, tfBuffer):

        # Give the cloud a chance to come in
        rospy.sleep(0.3)

        # Start timer
        start_time = time.time()
        
        ###### Get Point Cloud ######
        cloud = self.cloud
        height = cloud.height
        width = cloud.width

        ###### Get Transformation ######
        child = 'camera_color_frame'
        # child = 'pcl_frame'
        parent = 'base_link'

        try:
            trans = tfBuffer.lookup_transform(parent, child, rospy.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("Unable to access transformation!")
            trans = None

        ###### Create Transformation Matrix ######
        x = trans.transform.translation.x
        y = trans.transform.translation.y
        z = trans.transform.translation.z

        q0 = trans.transform.rotation.w
        q1 = trans.transform.rotation.x
        q2 = trans.transform.rotation.y
        q3 = trans.transform.rotation.z

        # First row of the rotation matrix
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)
        
        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)
        
        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1

        trans_matrix = np.array([[r00, r01, r02, x],
                                [r10, r11, r12, y],
                                [r20, r21, r22, z],
                                [  0,   0,   0, 1]])
        

        ###### Convert Point Cloud to Numpy ######
        pc_array = pointcloud2_to_array(cloud, squeeze=True)
        xyz_array = pointcloud2_to_xyz_array(cloud, remove_nans=False)

        ###### Transform PointCloud ######
        transformed_array = np.zeros((height, width, 3))
        for i in range(height):
            for ii in range(width):
                p_in = np.array([[xyz_array[i,ii,0]],[xyz_array[i,ii,1]],[xyz_array[i,ii,2]],[1]])
                if np.isnan(p_in).any():
                    transformed_array[i,ii,:] = p_in[0:3].transpose()
                else:
                    temp = np.matmul(trans_matrix, p_in)
                    p_out = np.array([temp[0], temp[1], temp[2]])
                    transformed_array[i,ii,:] = p_out.transpose()

        ###### Convert Numpy to Point Cloud ######
        transformed_pc_array = np.empty_like(pc_array, dtype=[('x', np.dtype('float32')), ('y', np.dtype('float32')), ('z', np.dtype('float32')), ('rgb', np.dtype('float32'))])
        for i in range(height):
            for ii in range(width):
                # print("pc_array element[()] length: ", len(pc_array[i,ii][()]))
                transformed_pc_array[i,ii][()][0] = transformed_array[i,ii,0]
                transformed_pc_array[i,ii][()][1] = transformed_array[i,ii,1]
                transformed_pc_array[i,ii][()][2] = transformed_array[i,ii,2]
                transformed_pc_array[i,ii][()][3] = pc_array[i,ii][()][3]

        self.transformed_cloud = array_to_pointcloud2(transformed_pc_array, stamp=cloud.header.stamp, frame_id=parent)

        # End timer
        print("Transformed Point Cloud in: ")
        print("--- %.2f seconds ---" % (time.time() - start_time))

    def publish(self):
        try:
            self.cloud_pub.publish(self.transformed_cloud)
        except:
            rospy.logwarn("No pointcloud received.")

def main():
    # Start the node
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('gpd_test')
    rate = rospy.Rate(10)

    # Create transform listener
    tfBuffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tfBuffer)

    # Initialize classes
    robot = Robot()
    cloud = Cloud()     # Before transform_cloud() is called, this will publish an empty cloud

    # 1) Open gripper and move to view pose (joint state)
    rospy.loginfo("Opening the gripper...")
    robot.reach_gripper_position(0)

    # position = 'gpd-view-overhead'
    position = 'gpd-view'
    rospy.loginfo("Reaching Named Joint Position: (" + position + ")...")
    robot.reach_named_joint_position(position, tolerance=0.01)

    # 2) Wait for user input
    input("Press Enter when ready to grasp. \n")

    # 3) Transform PointCloud2
    cloud.transform_cloud(tfBuffer)

    # 4) Run GPD
    # uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    # roslaunch.configure_logging(uuid)
    # launch = roslaunch.parent.ROSLaunchParent(uuid, ["/home/labuser/ros_ws/src/gpd_ros/launch/kinova.launch"])
    # launch.start()
    # # print("Waiting for grasp...")
    # while len(robot.grasps) <= 0:
    #     rate.sleep()
    # launch.shutdown()

    # package = 'gpd_ros'
    # executable = 'detect_grasps'
    # # cli_args = ['cloud_type:=0', 'cloud_topic:=/cloud_transformed', 'samples_topic:=""', 'config_file:=/home/labuser/code/gpd/cfg/ros_eigen_params.cfg', 'rviz_topic:=plot_grasps']
    # # args = roslaunch.rlutil.resolve_launch_arguments(cli_args)
    # node = roslaunch.core.Node(package, executable, name="detect_grasps", namespace='/my_gen3', output="screen")
    # launch = roslaunch.scriptapi.ROSLaunch()
    # rospy.set_param('cloud_type', 0)
    # rospy.set_param('cloud_topic', '/cloud_transformed')
    # rospy.set_param('samples_topic', "/home/labuser/code/gpd/cfg/ros_eigen_params.cfg")
    # rospy.set_param('rviz_topic', 'plot_grasps')
    # launch.start()
    # process = launch.launch(node)
    # rospy.sleep(10)
    # process.stop()

    # grasps = detect_grasps.run()

    # uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    # roslaunch.configure_logging(uuid)
    # cli_args = ['gpd_ros', 'detect_grasps', 'cloud_type:=0', 'cloud_topic:=/cloud_transformed', 'samples_topic:=""', 'config_file:=/home/labuser/code/gpd/cfg/ros_eigen_params.cfg', 'rviz_topic:=plot_grasps']
    # roslaunch_file = roslaunch.rlutil.resolve_launch_arguments(cli_args)
    # roslaunch_args = cli_args[2:]
    # launch_files = [(roslaunch_file, roslaunch_args)]
    # parent = roslaunch.parent.ROSLaunchParent(uuid, launch_files)
    # parent.start()
    # rospy.sleep(10)
    # parent.stop()

    # node = roslaunch.core.Node(package, executable, args=cli_args)
    # launch = roslaunch.scriptapi.ROSLaunch()
    # launch.start()
    # process = launch.launch(node)
    # print(process.is_alive())
    # rospy.sleep(5.0)
    # process.stop()

    # 5) Create grasp poses from GPD grasps
    poses = robot.grasp_to_pose(tfBuffer)
    print("Poses: ", poses)

    # 6) TODO: Remove infeasible grasps and choose best grasp
    grasp_pose = poses[0]

    # 7) Move to the selected grasp_pose
    robot.reach_cartesian_pose(pose=grasp_pose, tolerance=0.01, constraints=None)

    # 8) Close gripper, move to view pose, and open gripper
    robot.reach_gripper_position(0.65)
    position = 'gpd-view'
    robot.reach_named_joint_position(position, tolerance=0.01)
    robot.reach_gripper_position(0)

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
