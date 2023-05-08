#!/usr/bin/env python3

# Author: Jack Schultz

# To run this node in a given namespace with rosrun (for example 'my_gen3'), start a Kortex driver and then run : 
# rosrun kortex_examples example_moveit_trajectories.py __ns:=my_gen3

import rospy
import tf2_ros
from sensor_msgs.msg import PointCloud2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

import numpy as np
import open3d as o3d

from pyntcloud import PyntCloud
import pandas as pd

class transform_cloud():
    """gpd_test"""
    def __init__(self):
        # Some Initial Parameters
        self.cloud = None
        self.transformed_cloud = None

        # Initialize the node
        super(transform_cloud, self).__init__()
        # moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('transform_cloud')

        # Subscribers
        self.tfBuffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tfBuffer)
        rospy.Subscriber("/camera/depth_registered/points", PointCloud2, self.callback)

        # Publishers
        self.cloud_pub = rospy.Publisher('/cloud_transformed', PointCloud2, queue_size=10)

    # Subscriber Callbacks
    def callback(self, msg):
        self.cloud = msg
        # print("ORIGINAL POINT CLOUD:")
        # print(self.cloud.header)
        # print(self.cloud.height)
        # print(self.cloud.width)
        # print("-----------------------------\n")

        # Get transform
        # try:
        #     trans = self.tfBuffer.lookup_transform('base_link', 'pcl_frame', rospy.Time())
        #     self.transformed_cloud = do_transform_cloud(msg, trans)
        # except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        #     trans = None

        # print("TRANSFORMED POINT CLOUD:")
        # print(self.transformed_cloud.header)
        # print(self.transformed_cloud.height)
        # print(self.transformed_cloud.width)
        # print("-----------------------------\n")

    def publish(self):
        try:
            self.cloud_pub.publish(self.transformed_cloud)
        except:
            rospy.logwarn("No pointcloud received.")

    def spinOnce(self):
        r = rospy.Rate(100)
        r.sleep()

    def get_cloud(self):
        cloud = self.cloud
        return cloud

def main():
    #################### Pseudo Code ####################
    # 1) Subscribe to Point Cloud
    # 2) Get tf from camera frame to base frame
    # 3) Transform Point Cloud
    # 4) Publish transformed Point Cloud

    #####################################################

    # Initialize class
    run = transform_cloud()
        
    # cloud = run.get_cloud()
    # ply_point_cloud = o3d.data.EaglePointCloud()()
    # pcd = o3d.io.read_point_cloud(ply_point_cloud.path)
    # o3d.visualization.draw(pcd)

    # while not rospy.is_shutdown():

    #     run.publish()
    #     run.spinOnce()

if __name__ == '__main__':
    main()
