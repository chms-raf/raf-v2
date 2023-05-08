#!/usr/bin/env python3

# Author: Jack Schultz

# To run this node in a given namespace with rosrun (for example 'my_gen3'), start a Kortex driver and then run : 
# rosrun kortex_examples example_moveit_trajectories.py __ns:=my_gen3

import rospy
import tf2_ros
import numpy as np
from sensor_msgs.msg import PointCloud2
from sensor_msgs.point_cloud2 import read_points, PointField
import point_cloud2_functions as pc
from random import randint
import time

class transform_cloud():
    """gpd_test"""
    def __init__(self):
        # Some Initial Parameters
        self.cloud = None
        self.transformed_cloud = None

        # Initialize the node
        super(transform_cloud, self).__init__()
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

    # Print information if verbose is True
    verbose = False
    debug = False

    # Give the cloud a chance to come in
    rospy.sleep(0.3)

    # Start timer
    start_time = time.time()
    
    ###### Get Point Cloud ######
    if debug:
        input("Press Enter to get point cloud\n")
    print("Getting point cloud...")
    cloud = run.get_cloud()
    height = cloud.height
    width = cloud.width
    print('Done.\n')

    if verbose:
        print("############### Point Cloud ###############")
        print("Header: ", cloud.header)
        print("Height: ", cloud.height)
        print("Width: ", cloud.width)
        print("Fields: ", cloud.fields)
        print("Is_Bigendian: ", cloud.is_bigendian)
        print("Point_Step: ", cloud.point_step)
        print("Row_Step: ", cloud.row_step)
        print("Is_Dense: ", cloud.is_dense)
        print("Data Size (bytes): ", len(cloud.data))
        print("###########################################\n")

    ###### Get Transformation ######
    child = 'camera_color_frame'
    # child = 'pcl_frame'
    parent = 'base_link'

    if debug:
        input("Press Enter to get transform from '/" + child + "' to '/" + parent + "'\n")
    print("Getting tranformation from '/" + child + "' to '/" + parent + "'...")
    try:
        trans = run.tfBuffer.lookup_transform(parent, child, rospy.Time())
        print('Done.\n')
        if verbose:
            print('############### Transformation ###############')
            print(trans)
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        rospy.logwarn("Unable to access transformation!")
        trans = None

    ###### Create Transformation Matrix ######
    if debug:
        input("Press Enter to create transformation matrix out of transform\n")
    print("Creating transformation...")
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
    print('Done.\n')
    
    if verbose:
        print("############### Transformation Matrix ###############")
        print(trans_matrix)

    ###### Convert Point Cloud to Numpy ######
    if debug:
        input("Press Enter to convert PointCloud2 to numpy array\n")
    print("Converting Point Cloud to numpy...")
    pc_array = pc.pointcloud2_to_array(cloud, squeeze=True)
    xyz_array = pc.pointcloud2_to_xyz_array(cloud, remove_nans=False)
    print('Done.\n')

    ###### Transform PointCloud ######
    if debug:    
        input("Press Enter to transform Point Cloud from '/" + child + "' to '/" + parent + "'\n")
    print("Transforming Point Cloud...")
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
    print('Done.\n')

    ###### Print Some Random Cloud Samples ######
    if verbose:
        if debug:
            input("Press Enter to print some random cloud samples\n")
        print("Printing some random cloud samples..")
        for r in range(5):
            temp = np.nan
            while np.isnan(temp).any():
                idx = randint(0, height)
                idy = randint(0, width)
                temp = transformed_array[idx, idy, :]
            # print("Value at (" + str(idx) + ", " + str(idy) + "): ", temp)
            print("Original: ", xyz_array[idx, idy, :])
            print("Transformed: ", transformed_array[idx, idy, :])
            print("---")
        print('Done.\n')

    ###### Convert Numpy to Point Cloud ######
    if debug:
        input("Press Enter to convert transformed array back to PointCloud2\n")
    print("Converting array...")
    transformed_pc_array = np.empty_like(pc_array, dtype=[('x', np.dtype('float32')), ('y', np.dtype('float32')), ('z', np.dtype('float32')), ('rgb', np.dtype('float32'))])
    for i in range(height):
        for ii in range(width):
            # print("pc_array element[()] length: ", len(pc_array[i,ii][()]))
            transformed_pc_array[i,ii][()][0] = transformed_array[i,ii,0]
            transformed_pc_array[i,ii][()][1] = transformed_array[i,ii,1]
            transformed_pc_array[i,ii][()][2] = transformed_array[i,ii,2]
            transformed_pc_array[i,ii][()][3] = pc_array[i,ii][()][3]

    run.transformed_cloud = pc.array_to_pointcloud2(transformed_pc_array, stamp=cloud.header.stamp, frame_id=parent)
    print('Done.\n')

    if verbose:
        print("############### Transformed Point Cloud ###############")
        print("Header: ", run.transformed_cloud.header)
        print("Height: ", run.transformed_cloud.height)
        print("Width: ", run.transformed_cloud.width)
        print("Fields: ", run.transformed_cloud.fields)
        print("Is_Bigendian: ", run.transformed_cloud.is_bigendian)
        print("Point_Step: ", run.transformed_cloud.point_step)
        print("Row_Step: ", run.transformed_cloud.row_step)
        print("Is_Dense: ", run.transformed_cloud.is_dense)
        print("Data Size (bytes): ", len(run.transformed_cloud.data))
        print("###########################################\n")


    # End timer
    print("Runtime: ")
    print("--- %.2f seconds ---" % (time.time() - start_time))

    print("\n")
    print("Publishing transformed cloud...")

    while not rospy.is_shutdown():
        run.publish()
        run.spinOnce()

if __name__ == '__main__':
    main()
