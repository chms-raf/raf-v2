#!/usr/bin/env python3

# This script executes actions selected by the user

import rospy, sys, copy
import numpy as np
import cv2, time

from raf.msg import RafState, Selection, DetectionList, Detection, FaceDetection
from std_msgs.msg import String

import moveit_commander
import moveit_msgs.msg
import tf2_ros
from math import pi
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
from sensor_msgs.msg import PointCloud2, PointField, Image
from tf.transformations import quaternion_from_euler, euler_from_quaternion, quaternion_slerp
from math import sqrt, inf, degrees, radians

import tools.camtools as camtools
from argparse import Namespace
from kortex_api.autogen.client_stubs.VisionConfigClientRpc import VisionConfigClient
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.messages import DeviceConfig_pb2, VisionConfig_pb2
import random

class executeAction(object):
    def __init__(self):
        # Params
        self.raf_state = None
        self.action = None
        self.bridge = CvBridge()
        self.selection = Selection()
        self.pcloud = PointCloud2()
        self.face_detection = FaceDetection()
        self.depth_array = None
        self.vel = 1.0
        self.accel = 1.0
        self.g_vel = 0.9
        self.g_accel = 0.9
        self.home_pose = None
        self.mouth_trigger_time = 1.0       # Time required to open your mouth before food is released

        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(15)

        # Publishers
        self.pub_msg = rospy.Publisher('raf_message', String, queue_size=10)
        self.request_pub = rospy.Publisher('raf_state_request', String, queue_size=10)

        # Subscribers
        rospy.Subscriber("/raf_state", RafState, self.state_callback)
        rospy.Subscriber("/camera/depth_registered/points", PointCloud2, self.pc_callback)
        self.sub_depth = rospy.Subscriber("/camera/depth_registered/sw_registered/image_rect", Image, self.depth_callback)
        # rospy.Subscriber("/camera/color/image_raw", Image, self.callback)
        # rospy.Subscriber("/scene_camera/color/image_raw", Image, self.scene_callback)
        rospy.Subscriber("/arm_camera_detections", DetectionList, self.arm_detection_callback)
        rospy.Subscriber("/raf_action", String, self.action_callback)
        rospy.Subscriber("/raf_selection", Selection, self.selection_callback)
        rospy.Subscriber("/face_detection", FaceDetection, self.face_detection_callback)

        self.tfBuffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tfBuffer)

        # Point Cloud Stuff
        # prefix to the names of dummy fields we add to get byte alignment correct. this needs to not
        # clash with any actual field names
        self.DUMMY_FIELD_PREFIX = '__'

        # mappings between PointField types and numpy types
        self.type_mappings = [(PointField.INT8, np.dtype('int8')), (PointField.UINT8, np.dtype('uint8')), (PointField.INT16, np.dtype('int16')),
                        (PointField.UINT16, np.dtype('uint16')), (PointField.INT32, np.dtype('int32')), (PointField.UINT32, np.dtype('uint32')),
                        (PointField.FLOAT32, np.dtype('float32')), (PointField.FLOAT64, np.dtype('float64'))]
        self.pftype_to_nptype = dict(self.type_mappings)
        self.nptype_to_pftype = dict((nptype, pftype) for pftype, nptype in self.type_mappings)

        # sizes (in bytes) of PointField types
        self.pftype_sizes = {PointField.INT8: 1, PointField.UINT8: 1, PointField.INT16: 2, PointField.UINT16: 2,
                        PointField.INT32: 4, PointField.UINT32: 4, PointField.FLOAT32: 4, PointField.FLOAT64: 8}

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

    def init_camera(self):

        args = Namespace()
        args.ip = '137.148.209.35'
        args.password = 'admin'
        args.username = 'admin'

        with camtools.DeviceConnection.createTcpConnection(args) as router:

            self.device_manager = DeviceManagerClient(router)
            self.vision_config = VisionConfigClient(router)

            # example core
            self.vision_device_id = self.get_device_id()

            if self.vision_device_id != 0:
                self.set_camera_option('brightness', -1.0)
                self.set_camera_option('contrast', -2.0)
                self.set_camera_option('saturation', 0.0)

    def get_device_id(self):
        self.vision_device_id = 0
    
        # Getting all device routing information (from DeviceManagerClient service)
        all_devices_info = self.device_manager.ReadAllDevices()

        vision_handles = [ hd for hd in all_devices_info.device_handle if hd.device_type == DeviceConfig_pb2.VISION ]
        if len(vision_handles) == 0:
            print("Error: there is no vision device registered in the devices info")
        elif len(vision_handles) > 1:
            print("Error: there are more than one vision device registered in the devices info")
        else:
            handle = vision_handles[0]
            self.vision_device_id = handle.device_identifier
            print("Vision module found, device Id: {0}".format(self.vision_device_id))

        return self.vision_device_id
    
    def set_camera_option(self, option, value):
        sensor = VisionConfig_pb2.SENSOR_COLOR
        option_value = VisionConfig_pb2.OptionValue()
        option_value.sensor = sensor
        options = {'brightness': {'id': 2, 'name': 'OPTION_BRIGHTNESS', 'writable': True, 'min': -4.0, 'max': 4.0, 'step': 1.0, 'default': 0.0},
                    'contrast': {'id': 3, 'name': 'OPTION_CONTRAST', 'writable': True, 'min': -4.0, 'max': 4.0, 'step': 1.0, 'default': 0.0},
                    'saturation': {'id': 8, 'name': 'OPTION_SATURATION', 'writable': True, 'min': -4.0, 'max': 4.0, 'step': 1.0, 'default': 0.0}}
        option_value.option = options[option]['id']
        if value > options[option]['max'] or value < options[option]['min']:
            print(f"ERROR! Value must be between {options[option]['min']} and {options[option]['max']} at increments of {options[option]['step']}")
        option_value.value = value
        self.vision_config.SetOptionValue(option_value, self.vision_device_id)

    def focus_camera(self):
        args = Namespace()
        args.ip = '137.148.209.35'
        args.password = 'admin'
        args.username = 'admin'

        with camtools.DeviceConnection.createTcpConnection(args) as router:

            self.device_manager = DeviceManagerClient(router)
            self.vision_config = VisionConfigClient(router)

            # example core
            self.vision_device_id = self.get_device_id()

            if self.vision_device_id != 0:
                sensor_focus_action = VisionConfig_pb2.SensorFocusAction()
                sensor_focus_action.sensor = VisionConfig_pb2.SENSOR_COLOR

                u = random.randrange(0, 1280)
                v = random.randrange(0, 720)

                sensor_focus_action.focus_action = VisionConfig_pb2.FOCUSACTION_SET_FOCUS_POINT
                sensor_focus_action.focus_point.x = int(u)
                sensor_focus_action.focus_point.y = int(v)
                self.vision_config.DoSensorFocusAction(sensor_focus_action, self.vision_device_id)
                sensor_focus_action.focus_action = VisionConfig_pb2.FOCUSACTION_FOCUS_NOW
                self.vision_config.DoSensorFocusAction(sensor_focus_action, self.vision_device_id)

    def init_set_positions(self):
        # self.home_joints = [0.34554671604492154, -1.2907505571031779, -1.852130001189777, 0.24850590296763947, -1.7745574449579484, 1.5154674240937918]       # More to center
        self.home_joints = [0.032253489866490015, -1.3306542976124893, -1.8414975968537446, 0.10480068192728456, -1.7593289572126576, 1.5593387419951033]       # More to the left to avoid tablet for now
        self.feed_idle_joints = [2.513291966051138, -1.095005553821264, -1.3615879789381964, 0.109762700315077, -1.5953208446412184, 1.4780307028602755]
        self.cup_joints = [-0.5945805407643991, 1.106155676673208, -0.9047104433940882, -2.099077472221773, 1.8922959299084068, 2.50452404079427]             # More to the left to avoid tablet for now
        self.pre_feed_grasp_joints = [1.155544664879, -0.044854024343551124, -1.3250680509099393, 0.794122679128548, -2.016315746605925, -0.0640495568486239]
        # self.feed_grasp_joints = [1.5796610579573889, -0.016946226648392404, -1.018963782584013, 0.8076683153810178, -1.9335814515493057, 0.4386946700435371]
        self.feed_grasp_joints = [1.4378760925762786, -0.047248738795749645, -1.1988789563464968, 0.8093016986565907, -1.9336991632694867, 0.23040917408357622]
        self.pre_sip_joints = [0.886066851024934, 0.08391501584828466, -1.699199573488631, 1.1333802403907063, -2.093660602564552, -0.4261116333670767]
        # self.feed_sip_joints = [1.2175008439527668, -0.07879654506882883, -1.7244276984949511, 1.2870921727137363, -1.834673246560417, -0.1592021720682828]
        self.feed_sip_joints = [1.1288620544433325, -0.028720061827632648, -1.8476036926010782, 1.2115385589042975, -1.8585982868453614, -0.35643907819286014]
        # TODO: eat-idle position can use visual servoing to move arm so centroid of all detections is centered in cam frame

    def action_callback(self, msg):
        self.action = msg.data
        if self.raf_state.system_state == 'action':
            self.mapAction()
        else:
            rospy.logwarn("Action %s requested but system state is %s", self.action, self.raf_state.system_state)

    def selection_callback(self, msg):
        self.selection = msg

    def face_detection_callback(self, msg):
        self.face_detection = msg

    def state_callback(self, msg):
        self.raf_state = msg

    def pc_callback(self, msg):
        self.pcloud = msg

    def depth_callback(self, msg):
        self.depth_array = self.convert_depth_image(msg)

    def arm_detection_callback(self, msg):

        temp = self.sort_detections(msg)
        self.arm_detections = DetectionList()
        for i in range(len(temp)):
            self.arm_detections.class_ids.append(temp[i][0])
            self.arm_detections.class_names.append(temp[i][1])
            self.arm_detections.scores.append(temp[i][2])
            self.arm_detections.boxes.append(temp[i][3])
            self.arm_detections.masks.append(temp[i][4])

    def convert_depth_image(self, image_msg):
        if image_msg is not None:
            depth_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
            depth_array = np.array(depth_image, dtype=np.float32)
            return depth_array
        else:
            return None

    def sort_detections(self, msg):
        target = self.Out_transfer(msg.class_ids, msg.class_names, msg.scores, msg.boxes, msg.masks)

        # Sort by y-offset
        self.Sort_quick(target, 0, len(target)-1)

        return target
    
    def Out_transfer(self, class_id, class_name, score, box, mask):
        num = int(len(class_id))
        target = []

        for i in range(num):
            target.append([class_id[i], class_name[i], score[i], box[i], mask[i]])

        return target
    
    def partition(self, target, low, high):

        i = ( low-1 )
        arr = []
        arr = [target[w][3].y_offset for w in range(len(target))]

        pivot = arr[high]

        for j in range(low , high): 
    
            if   arr[j] <= pivot: 
            
                i = i+1 
                target[i],target[j] = target[j],target[i] 
    
        target[i+1],target[high] = target[high],target[i+1] 

        return ( i+1 )

    def Sort_quick(self, target, low, high):

        if low < high: 
            pi = self.partition(target, low, high) 
    
            self.Sort_quick(target, low, pi-1) 
            self.Sort_quick(target, pi+1, high)

    def get_detections(self):
        rospy.sleep(0.2)
        return self.arm_detections

    def get_selection(self):
        rospy.sleep(0.2)
        return self.selection
    
    def pixel_to_3d_point(self, pix_u, pix_v):
        '''This function converts pixels to 3D points relative to the camera's coordinate frame.
        It accepts either single pixel values or a list of pixels.
        A list of pixels as the input will return the average of the converted 3D points.'''

        pix_u = np.squeeze(pix_u)
        pix_v = np.squeeze(pix_v)

        try:
            if len(pix_u) != len(pix_v):
                rospy.ERROR("u and v must be the same length.")
        except:
            pix_u = [pix_u]
            pix_v = [pix_v]

        xyz_array = self.pointcloud2_to_xyz_array(self.pcloud, remove_nans=False)
        
        x_list = list()
        y_list = list()
        z_list = list()
        for u, v in zip(pix_u, pix_v):

            p = Point()
            p.x = xyz_array[v, u, 0]
            p.y = xyz_array[v, u, 1]
            p.z = xyz_array[v, u, 2]

            p_array = [p.x, p.y, p.z]

            # if points are meaningless, take average of surrounding points
            # TODO: could continually increase pad if still no meaningful values are found
            pad = 5
            pad_array = np.zeros(((2*pad)+1, (2*pad)+1, 3))
            if (np.isnan(p_array).any()) or (all(v == 0 for v in p_array)):
                pad_array[:,:,0] = xyz_array[v-pad:v+pad+1, u-pad:u+pad+1, 0]
                pad_array[:,:,1] = xyz_array[v-pad:v+pad+1, u-pad:u+pad+1, 1]
                pad_array[:,:,2] = xyz_array[v-pad:v+pad+1, u-pad:u+pad+1, 2]

                no_nans = pad_array[np.logical_not(np.isnan(pad_array))]
                temp = np.reshape(no_nans, (int(len(no_nans)/3), 3))

                p.x = np.mean(temp, axis=0)[0]
                p.y = np.mean(temp, axis=0)[1]
                p.z = np.mean(temp, axis=0)[2]

                p_array = [p.x, p.y, p.z]

                if not (np.isnan(p_array).any()) or not (all(v == 0 for v in p_array)):
                    x_list.append(p.x)
                    y_list.append(p.y)
                    z_list.append(p.z)
            else:
                x_list.append(p.x)
                y_list.append(p.y)
                z_list.append(p.z)

        x_list = np.array(x_list)[np.logical_not(np.isnan(x_list))]
        y_list = np.array(y_list)[np.logical_not(np.isnan(y_list))]
        z_list = np.array(z_list)[np.logical_not(np.isnan(z_list))]

        x_avg = np.mean(x_list)
        y_avg = np.mean(y_list)
        z_avg = np.mean(z_list)

        result = Point()
        result.x = x_avg
        result.y = y_avg
        result.z = z_avg

        return result
    
    def compute_mask_depth(self, mask):
        # Find only the coordinates of the object
        coord = cv2.findNonZero(mask)
        # print("coord.shape: ", coord.shape)

        # Get snapshot of pointcloud in xyz coordinates
        xyz_array = self.pointcloud2_to_xyz_array(self.pcloud, remove_nans=False)
        depth_list = xyz_array[coord[::, 0, 1], coord[::, 0, 0], 2]

        # Remove nans
        no_nans = depth_list[np.logical_not(np.isnan(depth_list))]
        avg_depth = np.mean(no_nans)

        return avg_depth
    
    def compute_sip_width(self, mask, gp):
        # Find only the coordinates of the object
        coord = np.squeeze(cv2.findNonZero(mask))

        # Return a list pixels in the mask which are at the same pixel height as our grasp point
        temp = [e for e in coord if e[1] == int(gp[1])]

        # Sort coordinate list by ascending x coordinate
        temp = np.array(sorted(temp, key=lambda k: k[0]))

        # Create window of points interior to edges of the cup
        left = np.squeeze(temp[5:15])
        right = np.squeeze(temp[len(temp)-10::])

        # For each point, convert it to 3D
        left_3D = self.pixel_to_3d_point(left[:,0], left[:,1])
        right_3D = self.pixel_to_3d_point(right[:,0], right[:,1])

        # Get distance between avg points
        item_width = self.compute_dist(left_3D, right_3D)

        return item_width

    def pointcloud2_to_xyz_array(self, cloud_msg, remove_nans=True):
        return self.get_xyz_points(self.pointcloud2_to_array(cloud_msg), remove_nans=remove_nans)
    
    def get_xyz_points(self, cloud_array, remove_nans=True, dtype=float):
        '''Pulls out x, y, and z columns from the cloud recordarray, and returns
            a 3xN matrix.
        '''
        # remove crap points
        if remove_nans:
            mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
            cloud_array = cloud_array[mask]
        
        # pull out x, y, and z values
        points = np.zeros(cloud_array.shape + (3,), dtype=dtype)
        points[...,0] = cloud_array['x']
        points[...,1] = cloud_array['y']
        points[...,2] = cloud_array['z']

        return points
    
    def pointcloud2_to_array(self, cloud_msg, squeeze=True):
        ''' Converts a rospy PointCloud2 message to a numpy recordarray 
        
        Reshapes the returned array to have shape (height, width), even if the height is 1.

        The reason for using np.fromstring rather than struct.unpack is speed... especially
        for large point clouds, this will be <much> faster.
        '''

        # construct a numpy record type equivalent to the point type of this cloud
        dtype_list = self.fields_to_dtype(cloud_msg.fields, cloud_msg.point_step)

        # parse the cloud into an array
        cloud_arr = np.frombuffer(cloud_msg.data, dtype_list)

        # remove the dummy fields that were added
        cloud_arr = cloud_arr[
            [fname for fname, _type in dtype_list if not (fname[:len(self.DUMMY_FIELD_PREFIX)] == self.DUMMY_FIELD_PREFIX)]]
        
        if squeeze and cloud_msg.height == 1:
            return np.reshape(cloud_arr, (cloud_msg.width,))
        else:
            return np.reshape(cloud_arr, (cloud_msg.height, cloud_msg.width))
        
    def fields_to_dtype(self, fields, point_step):
        '''Convert a list of PointFields to a numpy record datatype.
        '''
        offset = 0
        np_dtype_list = []
        for f in fields:
            while offset < f.offset:
                # might be extra padding between fields
                np_dtype_list.append(('%s%d' % (self.DUMMY_FIELD_PREFIX, offset), np.uint8))
                offset += 1

            dtype = self.pftype_to_nptype[f.datatype]
            if f.count != 1:
                dtype = np.dtype((dtype, f.count))

            np_dtype_list.append((f.name, dtype))
            offset += self.pftype_sizes[f.datatype] * f.count

        # might be extra padding between points
        while offset < point_step:
            np_dtype_list.append(('%s%d' % (self.DUMMY_FIELD_PREFIX, offset), np.uint8))
            offset += 1
            
        return np_dtype_list

    def mapAction(self):

        if self.action is not None and self.action != 'Cancel':
            if self.action == 'Grasp':
                self.grasp()
            elif self.action == 'Sip':
                self.sip()
            else:
                start_time = rospy.get_time()
                while (rospy.get_time() - start_time) < 5.0:
                    self.pub_msg.publish('Executing ' + self.action)
                    rospy.sleep(0.25)
                    self.pub_msg.publish('Executing ' + self.action + '.')
                    rospy.sleep(0.25)
                    self.pub_msg.publish('Executing ' + self.action + '..')
                    rospy.sleep(0.25)
                    self.pub_msg.publish('Executing ' + self.action + '...')
                    rospy.sleep(0.25)
                self.pub_msg.publish(self.action + ' complete.')
                rospy.sleep(1.0)
                self.request_pub.publish('to-gui-disable-selection')
                self.request_pub.publish('system-state-idle')
        else:
            self.pub_msg.publish('Error(3) - Invalid Action.')
            self.request_pub.publish('system-state-idle')

    def acquire_grasp_pose(self, det, c, m, a, b):

        mask = self.bridge.imgmsg_to_cv2(det.mask)

        # Compute grasp point
        gp = np.squeeze(np.mean(np.array([[m],[c]]), axis=0))
        # gp = np.squeeze(np.mean(np.array([[gp],[c]]), axis=0))

        # Find average depth value across object mask for z value
        avg_depth = self.compute_mask_depth(mask)

        # Get grasp point 3D coord
        p = self.pixel_to_3d_point(int(gp[0]), int(gp[1]))
        p_orig = self.transform_point(p, 'camera_color_frame', 'base_link')
        p.z = avg_depth
        p = self.transform_point(p, 'camera_color_frame', 'base_link')
        print("Original point: ", p_orig)
        print("New point: ", p)

        # Compute Angle
        vector_1 = m - c
        vector_2 = [1, 0]
        unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
        unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle = np.arccos(dot_product)
        theta = 90 - np.rad2deg(angle)

        # Compute gripper positions
        p_a = self.pixel_to_3d_point(int(a[0]), int(a[1]))
        p_a.z = avg_depth
        p_a = self.transform_point(p_a, 'camera_color_frame', 'base_link')

        p_b = self.pixel_to_3d_point(int(b[0]), int(b[1]))
        p_b.z = avg_depth
        p_b = self.transform_point(p_b, 'camera_color_frame', 'base_link')

        item_width = self.compute_dist(p_a, p_b)
        gripper_width = self.linear_map(0.0, 0.14, 1.0, 0.0, item_width)
        gripper_open = gripper_width - gripper_width*0.15
        gripper_close = gripper_width

        # Make sure gripper position within range
        gripper_max = 0.98
        gripper_min = 0.02
        if gripper_open < gripper_min:
            gripper_open = gripper_min
        elif gripper_open > gripper_max:
            gripper_open = gripper_max

        if gripper_close < gripper_min:
            gripper_close = gripper_min
        elif gripper_close > gripper_max:
            gripper_close = gripper_max

        # print("Item Width: ", item_width)
        # print("Gripper Width: ", gripper_width)
        # print("Gripper Pre-Close: ", gripper_open)
        # print("Gripper Close: ", gripper_close)
        # input("PRESS ENTER TO CONTINUE...")

        return p, theta, gripper_open, gripper_close
    
    def acquire_sip_pose(self, det, c):

        # box = det.box
        mask = self.bridge.imgmsg_to_cv2(det.mask)

        # Compute grasp point
        gp = c      # Centroid

        # Find average depth value across object mask for z value
        # TODO: Instead of whole mask, just take avg depth across window near target point
        avg_depth = self.compute_mask_depth(mask)

        # Get grasp point 3D coord
        p = self.pixel_to_3d_point(int(gp[0]), int(gp[1]))
        p_orig = self.transform_point(p, 'camera_color_frame', 'base_link')
        p.z = avg_depth
        p = self.transform_point(p, 'camera_color_frame', 'base_link')
        # print("Original point: ", p_orig)
        # print("New point: ", p)

        # Compute Angle
        # TODO: Might need this for complex cups

        # Compute gripper positions
        # TODO: computing cup width is not very reliable right now
        # item_width = self.compute_sip_width(mask, gp)
        # item_width = item_width*1.15        # Extra adjustment because of the window size when computing width
        item_width = 0.074
        gripper_width = self.linear_map(0.0, 0.14, 1.0, 0.0, item_width)
        # gripper_open = gripper_width - gripper_width*0.75
        gripper_open = 0.0
        gripper_close = gripper_width

        # Make sure gripper position within range
        gripper_max = 0.98
        gripper_min = 0.02
        if gripper_open < gripper_min:
            gripper_open = gripper_min
        elif gripper_open > gripper_max:
            gripper_open = gripper_max

        if gripper_close < gripper_min:
            gripper_close = gripper_min
        elif gripper_close > gripper_max:
            gripper_close = gripper_max

        # print("Item Width: ", item_width)
        # print("Gripper Width: ", gripper_width)
        # print("Gripper Pre-Close: ", gripper_open)
        # print("Gripper Close: ", gripper_close)
        # input("PRESS ENTER TO CONTINUE...")

        return p, gripper_open, gripper_close, item_width
    
    def linear_map(self, x1, x2, y1, y2, x):
        slope = 1.0 * (y2 - y1) / (x2 - x1)
        y = (slope * (x - x1)) +  y1
        return y
    
    def execute_grasp(self, p, theta, gripper_open, gripper_close):

        # Pre-close Gripper
        # gripper_open = 0.5
        success = self.move('gripper', gripper_open)

        # Rotate first
        P = self.get_cartesian_pose()
        e = euler_from_quaternion([P.orientation.x, P.orientation.y, P.orientation.z, P.orientation.w])
        current_angle = degrees(e[2])
        desired_angle = current_angle + theta
        q = quaternion_from_euler(e[0], e[1], radians(desired_angle))

        print("\nGrasp Angle Info:")
        print("theta: ", theta)
        print("Current angle: ", current_angle)
        print("Desired angle: ", desired_angle)

        P.position.x = p.x + 0.01
        P.position.y = p.y + 0.01
        P.position.z = p.z + 0.1
        P.orientation.x = q[0]
        P.orientation.y = q[1]
        P.orientation.z = q[2]
        P.orientation.w = q[3]

        # waypoints = self.compute_waypoints(P)
        success = self.move('pose', P, tolerance=0.01, vel=0.8, accel=0.8, attempts=10, time=5.0, constraints=None)
        if success:
            print("Done.")
        else:
            print("Rotation failed.")
            self.reset()

        # Move to grasp position
        P = self.get_cartesian_pose()

        # TODO: Get depth map again in this pose to fine-tune depth.
        # It's possible the camera will be too close for an accurate depth map, though.
        P.position.z = p.z + 0.0325

        # Orientation constraint (we want the end effector to stay the same orientation)
        # constraints = moveit_msgs.msg.Constraints()
        # # print("\n### CONSTRAINTS ###")
        # # print(constraints)
        # # print("#####################################\n")
        # orientation_constraint = moveit_msgs.msg.OrientationConstraint()
        # orientation_constraint.header.frame_id = 'base_link'
        # # orientation_constraint.link_name = 'end_effector_link'
        # orientation_constraint.link_name = 'robotiq_arg2f_base_link'
        # orientation_constraint.absolute_x_axis_tolerance = 0.05
        # orientation_constraint.absolute_y_axis_tolerance = 0.05
        # orientation_constraint.absolute_z_axis_tolerance = 0.05
        # orientation_constraint.weight = 1.0

        # print("\n### ORIENTATION_CONSTRAINT ###")
        # print(orientation_constraint)
        # print("#####################################\n")

        # orientation_constraint.orientation = P.orientation
        # print("\n### ORIENTATION_CONSTRAINT ###")
        # print(orientation_constraint)
        # print("#####################################\n")

        # constraints.orientation_constraints.append(orientation_constraint)
        # print("\n### CONSTRAINTS ###")
        # print(constraints)
        # print("#####################################\n")

        # print("\n### Desired Pose ###")
        # print(P)
        # print("#####################################\n")

        waypoints = self.compute_waypoints(P)
        success = self.move('path', waypoints, tolerance=0.01, vel=0.05, accel=0.05, attempts=30, time=10.0, constraints=None)
        if success:
            print("Done.")
        else:
            print("Grasping failed.")
            self.reset()

        # Close Gripper
        # gripper_close = 0.9
        success = self.move('gripper', gripper_close)

    def execute_sip(self, p, gripper_open, gripper_close, item_width):

        # Pre-close Gripper
        # gripper_open = 0.5
        # success = self.move('gripper', gripper_open)

        # Rotate first
        # TODO: Might need this for complex cups

        # Move to grasp position
        P = self.get_cartesian_pose()

        # TODO: Get depth map again in this pose to fine-tune depth.
        # It's possible the camera will be too close for an accurate depth map, though.
        # P.position.x = p.x + item_width/4       # Good guess for symmetric cups
        P.position.x = p.x
        P.position.y = p.y - 0.03           # TODO: For some reason need this. Unclear for now
        P.position.z = p.z + 0.025              # TODO: Height adjust to avoid table. Better method needed

        print("MOVING TO: ", P)

        # waypoints = self.compute_waypoints(P)
        success = self.move('pose', P, tolerance=0.05, vel=0.3, accel=0.3, attempts=100, time=50.0, constraints=None)
        if success:
            print("Done.")
        else:
            print("Grasping failed.")
            self.reset()

        # Close Gripper
        # gripper_close = 0.9
        success = self.move('gripper', gripper_close)

        return success, P
    
    def order_points(self, pts):
        '''This function sorts the corners of the mask box points to start at the top left and continue clockwise'''
        # rect = np.zeros((4, 2), dtype = "float32")

        # Let's first sort the points by y, and then by x
        box = np.array(sorted(pts, key=lambda k: [k[1], k[0]]))

        # Let's create a matrix to hold the sorted points and their difference to the first point
        mat = np.zeros((4, 3), dtype = "float32")

        # Now we take the difference between the points and the first point
        for i in range(len(box)):
            mat[i, 0:2] = box[i]
            mat[i, 2] = np.linalg.norm(box[0] - box[i])

        # Now, let's sort our new matrix according to the difference
        sorted_mat = np.array(sorted(mat, key=lambda k: k[2]))

        # Finally, slice the matrix to return just the box points
        sorted_box = sorted_mat[:, 0:2]

        # return the ordered coordinates
        return sorted_box
    
    def compute_centroid(self, mask):

        mask = self.bridge.imgmsg_to_cv2(mask)
        _, thresh = cv2.threshold(mask, 0, 255, 0)

        cntrs = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

        # TODO: Use moment of inertia instead to compute orientation
        rotrect = cv2.minAreaRect(cntrs[0])
        box = cv2.boxPoints(rotrect)
        box = np.int0(box)

        # Reorder the box points
        box = self.order_points(box)
        box = np.int0(box)

        # Compute important points
        c = np.mean(box, axis=0)
        m = np.mean(box[2:], axis=0)
        a = box[2]
        b = box[3]
        return c, m, a, b
    
    def tf_from_trans(self, trans):
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
        
        return trans_matrix
    
    def transform_point(self, p, from_frame, to_frame):
        trans = self.get_tf(to_frame, from_frame)
        T = self.tf_from_trans(trans)
        
        P = np.array([[p.x],[p.y],[p.z], [1]])
        temp = np.matmul(T, P)
        temp = temp[:][0:-1]

        p = Point()
        p.x = temp[0][0]
        p.y = temp[1][0]
        p.z = temp[2][0]

        return(p)
    
    def search_detections(self, P):
        det = Detection()
        # Get all detections
        det_list = self.get_detections()
        selection = self.selection
        # TODO: Change detection list message type to just be a list of detections
        # Could then instead just do "for det in detections" - better for looping
        idx = None
        cent = None
        mid = None
        prev_d = inf
        for i in range(len(det_list.class_ids)):
            if det_list.class_ids[i] == selection.class_id:
                c, m, a, b = self.compute_centroid(det_list.masks[i])
                p = self.pixel_to_3d_point(int(c[0]), int(c[1]))
                p = self.transform_point(p, 'camera_color_frame', 'base_link')
                d = self.compute_dist(P, p)
                print("######## DISTANCE STUFF ############")
                print(f"Distance to detection {i}: {d:2.2f} meters")
                print("####################################")
                if d < prev_d:
                    idx = i
                    cent = c
                    mid = m
                    left = a
                    right = b
                    prev_d = d

        det.box = det_list.boxes[idx]
        det.class_id = det_list.class_ids[idx]
        det.class_names = det_list.class_names[idx]
        det.score = det_list.scores[idx]
        det.mask = det_list.masks[idx]

        return det, cent, mid, left, right
    
    def compute_dist(self, p1, p2):
        '''Compute Euclidean distance between two points'''
        dist = sqrt(((p1.x - p2.x)**2) + ((p1.y - p2.y)**2) + ((p1.z - p2.z)**2))
        return dist

    def reset(self):
        '''When an anomaly is detected, reset everything and return home'''
        self.request_pub.publish('reset')
        rospy.sleep(0.1)

        print("Anomaly detected. Attempting reset...")
        success = self.move('joint', self.home_joints, tolerance=0.01, vel=1.0, accel=0.9, attempts=10, time=5.0, constraints=None)
        if success:
            print("Reset successful.")
        else:
            rospy.logerr("Reset after anomaly failed. Quitting Program.")
            sys.exit()

    def mouth_trigger(self):
        '''This function continuously checks the mouth detection
        and returns a trigger signal if it is open longer than
        a set duration'''

        done = False
        timer = None
        while not done:
            self.loop_rate.sleep()
            mouth_open = self.face_detection.mouth_open
            if timer is None:
                if mouth_open:
                    start_time = time.time()
                    timer = 0
            elif timer < self.mouth_trigger_time:
                if not mouth_open:
                    timer = None
                else:
                    timer = time.time() - start_time
            elif timer >= self.mouth_trigger_time:
                done = True
            else:
                done = False
                timer = None
    
    def grasp(self):
        print("\n---------- Beginning Grasp ----------")

        self.pub_msg.publish('Wait for grasp.')
        rospy.sleep(0.1)

        # Get center of selection mask in pixels [u,v]
        selection = self.get_selection()
        print("Selected Pixel: [" + str(int(selection.centroid.x)) + ", " + str(int(selection.centroid.y)) + "]")

        # Get 3D location of point in camera frame in meters [x,y,z]
        p = self.pixel_to_3d_point(int(selection.centroid.x), int(selection.centroid.y))
        print(f"Point (cam frame): {p.x:1.2f}, {p.y:1.2f}, {p.z:1.2f}")

        # Get 3D location of point in base frame in meters [x,y,z]
        p = self.transform_point(p, 'camera_color_frame', 'base_link')
        print(f"Point (base frame): {p.x:1.2f}, {p.y:1.2f}, {p.z:1.2f}")

        # Construct overhead pose
        # TODO: Maybe this step can be skipped to save time in the future
        P_overhead = self.get_cartesian_pose()
        P_overhead.position.x = p.x - .05
        P_overhead.position.y = p.y
        P_overhead.position.z = p.z + .25
        q = quaternion_from_euler(pi, 0, pi/2)
        P_overhead.orientation.x = q[0]
        P_overhead.orientation.y = q[1]
        P_overhead.orientation.z = q[2]
        P_overhead.orientation.w = q[3]

        # Compute waypoints
        # waypoints = self.compute_waypoints(P_overhead)
        # success = self.move('path', waypoints, tolerance=0.01, vel=1.0, accel=0.9, attempts=10, time=5.0, constraints=None)
        success = self.move('pose', P_overhead, tolerance=0.01, vel=1.0, accel=1.0, attempts=10, time=5.0, constraints=None)
        if success:
            print("Done.")
        else:
            print("Move to Overhead failed.")
            self.reset()

        rospy.sleep(2.0)

        # Search detections for one containing centroid
        det, c, m, a, b = self.search_detections(p)

        # Acquire item pose
        pos, theta, gripper_open, gripper_close = self.acquire_grasp_pose(det, c, m, a, b)

        # Acquire item
        success = self.execute_grasp(pos, theta, gripper_open, gripper_close)

        rospy.sleep(2.0)

        # Retract
        P = self.get_cartesian_pose()
        P.position.z += 0.1
        # waypoints = self.compute_waypoints(P)
        success = self.move('pose', P, tolerance=0.01, vel=0.5, accel=0.5, attempts=10, time=10.0, constraints=None)

        # Move to feed idle position
        success = self.move('joint', self.feed_idle_joints, tolerance=0.01, vel=1.0, accel=0.9, attempts=10, time=5.0, constraints=None)
        if success:
            print("Done.")
        else:
            print("Move to Feed_Idle failed.")
            self.reset()

        # Move to pre-feed position
        success = self.move('joint', self.pre_feed_grasp_joints, tolerance=0.01, vel=0.8, accel=0.9, attempts=10, time=10.0, constraints=None)
        if success:
            print("Done.")
        else:
            print("Move to Pre-feed_grasp failed.")

        ### VISUAL SERVOING ###
        # Move to pre-sip position
        # success = self.move('joint', self.pre_sip_joints, tolerance=0.01, vel=0.8, accel=0.8, attempts=10, time=10.0, constraints=None)
        # if success:
        #     print("Done.")
        # else:
        #     print("Move to Pre-feed_sip failed.")
        #     self.reset()

        # # Enable face detection
        # self.request_pub.publish('disable-arm-detections')
        # rospy.sleep(0.1)

        # self.request_pub.publish('enable-face-detection')
        # rospy.sleep(0.1)

        # # Enable visual servoing
        # self.request_pub.publish('enable-visual-servoing')
        # rospy.sleep(0.1)

        # self.pub_msg.publish('Take a bite when ready.')
        # rospy.sleep(0.1)

        # # This will wait until mouth trigger is detected
        # self.mouth_trigger()

        # self.request_pub.publish('disable-visual-servoing')
        # rospy.sleep(0.1)
        
        # self.pub_msg.publish('Mouth Open Detected!')
        # rospy.sleep(1.0)

        # self.pub_msg.publish('Releasing Food.')
        # rospy.sleep(0.1)

        #######################

        # Move to feed position
        success = self.move('joint', self.feed_grasp_joints, tolerance=0.01, vel=0.5, accel=0.5, attempts=30, time=10.0, constraints=None)
        if success:
            print("Done.")
        else:
            print("Move to Feed_grasp failed.")

        rospy.sleep(2.0)

        # Open Gripper
        success = self.move('gripper', 0)

        rospy.sleep(0.5)

        self.request_pub.publish('disable-face-detection')
        rospy.sleep(0.1)

        # Move to pre-feed position
        success = self.move('joint', self.pre_feed_grasp_joints, tolerance=0.01, vel=0.5, accel=0.5, attempts=10, time=5.0, constraints=None)
        if success:
            print("Done.")
        else:
            print("Move to Pre-feed_grasp failed.")
            self.reset()

        # Return to home
        print("\nMoving to Home position...")
        # waypoints = self.compute_waypoints(self.home_pose)
        success = self.move('pose', self.home_pose, tolerance=0.01, vel=1.0, accel=0.9, attempts=10, time=5.0, constraints=None)
        if success:
            print("Done.")
        else:
            print("Move to Home failed.")
            self.reset()

        rospy.sleep(0.1)

        self.pub_msg.publish(self.action + ' complete.')
        rospy.sleep(1.0)
        self.request_pub.publish('to-gui-disable-selection')
        rospy.sleep(0.1)
        self.request_pub.publish('system-state-idle')

        # Focus Camera
        print("Focusing camera...")
        rospy.sleep(1.0)
        self.focus_camera()
        print("Done.")

    def sip(self):
        print("\n---------- Beginning Sip ----------")
        
        # Get center of selection mask in pixels [u,v]
        selection = self.get_selection()
        print("Selected Pixel: [" + str(int(selection.centroid.x)) + ", " + str(int(selection.centroid.y)) + "]")

        # Get 3D location of point in camera frame in meters [x,y,z]
        p = self.pixel_to_3d_point(int(selection.centroid.x), int(selection.centroid.y))
        print(f"Point (cam frame): {p.x:1.2f}, {p.y:1.2f}, {p.z:1.2f}")

        # Get 3D location of point in base frame in meters [x,y,z]
        p = self.transform_point(p, 'camera_color_frame', 'base_link')
        print(f"Point (base frame): {p.x:1.2f}, {p.y:1.2f}, {p.z:1.2f}")

        # Move to pre-defined cup pose
        # TODO: Need a better method for this. Currently can't plan to it
        success = self.move('joint', self.cup_joints, tolerance=0.01, vel=1.0, accel=1.0, attempts=20, time=10.0, constraints=None)
        if success:
            print("Done.")
        else:
            print("Move to Cup_joints failed.")
            self.reset()

        # Construct pre-grasp pose
        # TODO: Maybe this step can be skipped to save time in the future
        P_pregrasp = self.get_cartesian_pose()
        P_pregrasp.position.x = p.x + .03
        P_pregrasp.position.y = p.y + .2
        P_pregrasp.position.z = p.z + .05
        # Orientation: If looking from the gripper's forward perspective, x is to the left, y is up, and z is pointing away
        thetaX = radians(110)
        thetaY = radians(0)
        thetaZ = radians(0)
        q = quaternion_from_euler(thetaX, thetaY, thetaZ)        # thetaX, thetaY, thetaZ
        P_pregrasp.orientation.x = q[0]
        P_pregrasp.orientation.y = q[1]
        P_pregrasp.orientation.z = q[2]
        P_pregrasp.orientation.w = q[3]

        # Move to pre-grasp pose
        print("MOVING TO: ", P_pregrasp)
        success = self.move('pose', P_pregrasp, tolerance=0.01, vel=1.0, accel=1.0, attempts=10, time=5.0, constraints=None)
        if success:
            print("Done.")
        else:
            print("Move to Pre-grasp failed.")
            self.reset()

        rospy.sleep(2.0)

        # Search detections for one containing centroid
        det, c, _, _, _ = self.search_detections(p)

        # Acquire item pose
        pos, gripper_open, gripper_close, item_width = self.acquire_sip_pose(det, c)

        # Acquire item
        success, sip_pose = self.execute_sip(pos, gripper_open, gripper_close, item_width)

        rospy.sleep(2.0)

        # Retract
        P = self.get_cartesian_pose()
        P.position.y += 0.1
        P.position.z += 0.05
        # waypoints = self.compute_waypoints(P)
        success = self.move('pose', P, tolerance=0.01, vel=0.5, accel=0.5, attempts=10, time=10.0, constraints=None)

        # Move to feed idle position
        success = self.move('joint', self.feed_idle_joints, tolerance=0.01, vel=1.0, accel=0.9, attempts=10, time=5.0, constraints=None)
        if success:
            print("Done.")
        else:
            print("Move to Feed_Idle failed.")
            self.reset()

        # Move to pre-sip position
        success = self.move('joint', self.pre_sip_joints, tolerance=0.01, vel=1.0, accel=0.9, attempts=10, time=5.0, constraints=None)
        if success:
            print("Done.")
        else:
            print("Move to Pre-sip failed.")
            self.reset()

        # Move to sip position
        success = self.move('joint', self.feed_sip_joints, tolerance=0.01, vel=0.25, accel=0.25, attempts=10, time=10.0, constraints=None)
        if success:
            print("Done.")
        else:
            print("Move to Feed_sip failed.")
            self.reset()

        rospy.sleep(5.0)

        # Move to pre-sip position
        success = self.move('joint', self.pre_sip_joints, tolerance=0.01, vel=1.0, accel=0.9, attempts=10, time=5.0, constraints=None)
        if success:
            print("Done.")
        else:
            print("Move to Pre-sip failed.")
            self.reset()

        # Move to feed idle position
        success = self.move('joint', self.feed_idle_joints, tolerance=0.01, vel=1.0, accel=0.9, attempts=10, time=5.0, constraints=None)
        if success:
            print("Done.")
        else:
            print("Move to Feed_Idle failed.")
            self.reset()

        # Move to home
        success = self.move('joint', self.home_joints, tolerance=0.01, vel=1.0, accel=0.9, attempts=10, time=5.0, constraints=None)
        if success:
            print("Done.")
        else:
            print("Move to Home failed.")
            self.reset()

        # Move to cup pre-place pose
        success = self.move('joint', self.cup_joints, tolerance=0.01, vel=1.0, accel=1.0, attempts=20, time=10.0, constraints=None)
        if success:
            print("Done.")
        else:
            print("Move to Cup_joints failed.")
            self.reset()

        # Move to cup place pose
        sip_pose.position.z += 0.05
        success = self.move('pose', sip_pose, tolerance=0.01, vel=1.0, accel=0.9, attempts=10, time=5.0, constraints=None)
        if success:
            print("Done.")
        else:
            print("Move to Cup pre-place pose failed.")
            self.reset()

        sip_pose.position.z -= 0.05
        success = self.move('pose', sip_pose, tolerance=0.01, vel=1.0, accel=0.9, attempts=10, time=5.0, constraints=None)
        if success:
            print("Done.")
        else:
            print("Move to Cup place pose failed.")
            self.reset()

        # Open Gripper
        success = self.move('gripper', 0)

        # Retract
        sip_pose.position.y += 0.1
        sip_pose.position.z += 0.05
        success = self.move('pose', sip_pose, tolerance=0.01, vel=1.0, accel=0.9, attempts=10, time=5.0, constraints=None)
        if success:
            print("Done.")
        else:
            print("Move to retract failed.")
            self.reset()

        rospy.sleep(0.1)

        # Return to home
        print("\nMoving to Home position...")
        # waypoints = self.compute_waypoints(self.home_pose)
        success = self.move('joint', self.home_joints, tolerance=0.01, vel=1.0, accel=0.9, attempts=10, time=5.0, constraints=None)
        if success:
            print("Done.")
        else:
            print("Move to Home failed.")
            self.reset()

        rospy.sleep(0.1)

        self.pub_msg.publish(self.action + ' complete.')
        rospy.sleep(1.0)
        self.request_pub.publish('to-gui-disable-selection')
        rospy.sleep(0.1)
        self.request_pub.publish('system-state-idle')

        # Focus Camera
        print("Focusing camera...")
        rospy.sleep(1.0)
        self.focus_camera()
        print("Done.")

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
        return arm_group.get_current_pose().pose

    def get_tf(self, parent, child):
        try:
            trans = self.tfBuffer.lookup_transform(parent, child, rospy.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            trans = None
        return trans

    def spinOnce(self):
        r = rospy.Rate(15)
        r.sleep()

def main():
    """ RAF Execute Action """
    rospy.init_node("raf_execute_action", anonymous=True)
    run = executeAction()
    run.init_set_positions()
    run.init_camera()

    success = run.success
    if success:
        print("\nMoving to home position...")
        success = run.move('joint', run.home_joints, tolerance=0.01, vel=1.0, accel=0.9, attempts=10, time=5.0, constraints=None)
        run.home_pose = run.get_cartesian_pose()
        success = run.move('gripper', 0)
        if success:
            print("Done.")
        else:
            print("Move to Home failed.")

    # Focus Camera
    print("Focusing camera...")
    rospy.sleep(1.0)
    run.focus_camera()
    print("Done.")

    rospy.spin()

if __name__ == '__main__':
    sys.exit(main())