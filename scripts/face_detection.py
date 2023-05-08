#!/usr/bin/env python3
import rospy
import sys, time, cv2
import numpy as np
import face_alignment
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2, PointField
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
from skimage.measure import EllipseModel
from skimage import io
from collections import deque

from raf.msg import RafState, FaceDetection
from tf import TransformBroadcaster, transformations

class FKD(object):
    def __init__(self):
        # Params
        self.image = None
        self.br = CvBridge()
        self.raf_message = ""
        self.raf_state = None
        self.pcloud = None

        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(15)

        # Publishers
        self.pub = rospy.Publisher('face_detection_image', Image, queue_size=10)
        self.face_pub = rospy.Publisher('face_detection', FaceDetection, queue_size=10)

        self.tf_pub = TransformBroadcaster()

        # Subscribers
        rospy.Subscriber("/camera/color/image_raw", Image, self.callback)
        rospy.Subscriber("/raf_state", RafState, self.state_callback)
        rospy.Subscriber("/camera/depth_registered/points", PointCloud2, self.pc_callback)

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

    def callback(self, msg):
        self.image = self.convert_to_cv_image(msg)
        self._header = msg.header

    def pc_callback(self, msg):
        self.pcloud = msg

    def state_callback(self, msg):
        self.raf_state = msg

    def get_img(self):
        result = self.image
        return result

    def convert_to_cv_image(self, image_msg):

        if image_msg is None:
            return None

        self._width = image_msg.width
        self._height = image_msg.height
        channels = int(len(image_msg.data) / (self._width * self._height))

        encoding = None
        if image_msg.encoding.lower() in ['rgb8', 'bgr8']:
            encoding = np.uint8
        elif image_msg.encoding.lower() == 'mono8':
            encoding = np.uint8
        elif image_msg.encoding.lower() == '32fc1':
            encoding = np.float32
            channels = 1

        cv_img = np.ndarray(shape=(image_msg.height, image_msg.width, channels),
                            dtype=encoding, buffer=image_msg.data)

        if image_msg.encoding.lower() == 'mono8':
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        else:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

        return cv_img

    def mouth_open(self, Points):
        mouth_points = Points[48:60]

        ell = EllipseModel()
        ell.estimate(mouth_points)

        xc, yc, a, b, theta = ell.params

        temp = [a, b]
        a = max(temp)
        b = min(temp)

        theta = np.rad2deg(theta)
        if theta > 85:
            theta = theta - 90

        ratio = a / b

        # print("center = ",  (xc, yc))
        # print("angle of rotation = ",  theta)
        # print("axes = ", (a,b))
        # print(("Ratio = ", ratio))

        # print(("\n-------------------\n"))

        if ratio > .5 and ratio < 2.0:
            mouth_open = True
        else:
            mouth_open = False

        return mouth_open, xc, yc, theta, a, b
    
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

        try:
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
        except:
            result = Point()

        return result
    
    def compute_mouth_pose(self, u, v):
        '''This function computes the 3D mouth pose relative to the camera's color frame'''
        mouth = self.pixel_to_3d_point(u, v)
        return [mouth.x, mouth.y, mouth.z]
    
    def publish(self, img, face, mouth_pose):
        self.pub.publish(img)
        self.face_pub.publish(face)

        if mouth_pose is not None:
            if not np.isnan(np.array(mouth_pose)).any():
                try:
                    self.tf_pub.sendTransform((mouth_pose[0], mouth_pose[1], mouth_pose[2]),
                                        transformations.quaternion_from_euler(0, 0, 0),
                                        rospy.Time.now(),
                                        "mouth",
                                        "camera_color_frame")
                    self.tf_pub.sendTransform((0.05, -0.1, -0.5),          # Slightly off-center to avoid issues, original (0, -0.6, -0.5)
                                    transformations.quaternion_from_euler(0, 0, 0),
                                    rospy.Time.now(),
                                    "desired_cam_frame",
                                    "mouth")
                except:
                    rospy.logwarn("Error broadcasting mouth tf.")
                    pass

        self.loop_rate.sleep()

def main():
    """ Face Detection """
    rospy.init_node("face_detection", anonymous=True)
    bridge = CvBridge()
    start_time = time.time()
    image_counter = 0

    # TODO: Make changeable parameters in the launch file such as det type, window size, etc.
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda', face_detector='sfd')

    # Run forward pass on sample image first. This initializes and loads the model, speeding up susequent inference
    print("Running warmup inference...")
    sample_img = io.imread('../img/face.jpg')
    for i in range(3):
        print("Detecting face instance " + str(i+1) + "...")
        _ = fa.get_landmarks(sample_img)[-1]
    print("Done.")

    run = FKD()
    numFace = None
    mouth_open = False
    mouth_open_msg = False
    mouth_timeout = 3.0
    movingAverageWindow = 1
    mouth_movingAverageWindow = 5

    timer = None

    radius = 3
    thickness = -1

    # Might be faster to use a deque so I can popleft()
    # I think popleft is O(1) runtime and list.append() is O(n)
    # Clarification: deques are better for appending and popping from beginning and end
    #                lists are better for accessing by index 
    # In this case, I shouldn't need to acess by index so a deque is better
    # xc_list = list()
    # yc_list = list()
    # theta_list = list()
    # a_list = list()
    # b_list = list()
    xc_list = deque()
    yc_list = deque()
    theta_list = deque()
    a_list = deque()
    b_list = deque()

    mpx_list = deque()
    mpy_list = deque()
    mpz_list = deque()

    face_msg = FaceDetection()
    mouth_pose = None
    
    while not rospy.is_shutdown():
        # Get images
        img = run.get_img()

        if img is None:
            continue

        if run.raf_state is None:
            continue

        if run.raf_state.enable_face_detections:

            try:
                det = fa.get_landmarks(img)[-1]
                numFace = 1         # TODO: Update to detect multiple faces
            except:
                numFace = None
            
            # det_type = collections.namedtuple('prediction_type', ['slice', 'color'])
            # det_types = {'face': det_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
            #                 'eyebrow1': det_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
            #                 'eyebrow2': det_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
            #                 'nose': det_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
            #                 'nostril': det_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
            #                 'eye1': det_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
            #                 'eye2': det_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
            #                 'lips': det_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
            #                 'teeth': det_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
            #                 }

            if numFace is None:
                continue

            Points = det.astype(int)

            try:
                mouth_open, xc, yc, theta, a, b = run.mouth_open(Points)
            except:
                continue

            xc_list.append(xc)
            yc_list.append(yc)
            theta_list.append(theta)
            a_list.append(a)
            b_list.append(b)
            if image_counter > movingAverageWindow:
                # xc_list.pop(0)
                # yc_list.pop(0)
                # theta_list.pop(0)
                # a_list.pop(0)
                # b_list.pop(0)
                xc_list.popleft()
                yc_list.popleft()
                theta_list.popleft()
                a_list.popleft()
                b_list.popleft()

            mean_xc = np.mean(xc_list)
            mean_yc = np.mean(yc_list)
            mean_theta = np.mean(theta_list)
            mean_a = np.mean(a_list)
            mean_b = np.mean(b_list)

            # mean_theta  = 0

            # print(mean_theta)

            # If mouth is open for 1 second it will automatically set the mouth to closed
            # if timer is None:
            #     if mouth_open:
            #         start_time = time.time()
            #         timer = 0
            # elif timer < mouth_timeout:
            #     if not mouth_open:
            #         timer = None
            #     else:
            #         timer = time.time() - start_time
            # else:
            #     mouth_open = False
            #     timer = None

            # Compute mouth pose
            mouth_pose = run.compute_mouth_pose(int(mean_xc), int(mean_yc))

            # Filter mouth pose based on moving average window
            # mpx_list.append(mouth_pose[0])
            # mpy_list.append(mouth_pose[1])
            # mpz_list.append(mouth_pose[2])
            # if image_counter > mouth_movingAverageWindow:
            #     mpx_list.popleft()
            #     mpy_list.popleft()
            #     mpz_list.popleft()

            # mean_mpx = np.mean(mpx_list)
            # mean_mpy = np.mean(mpy_list)
            # mean_mpz = np.mean(mpz_list)

            # mouth_pose = [mean_mpx, mean_mpy, mean_mpz]

            if mouth_open:
                mouth_color = (0, 255, 0)
            else:
                mouth_color = (0, 0, 255)

            # Draw individual face keypoints
            for i in range(60):
                point = Points[i]

                if i > 47:
                    # Mouth Color
                    color = mouth_color
                    radius = 3
                else:
                    color = (255, 0, 0)
                    radius = 2

                cv2.circle(img, tuple(point), radius, color, thickness)
                # cv2.putText(img, str(i), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, .2, (0,255,255), 1, cv2.LINE_AA)

            # Build face detection message
            print("Theta: ", mean_theta)
            face_msg.num_face = numFace
            face_msg.mouth_open = mouth_open
            face_msg.mouth_x = mean_xc.item()
            face_msg.mouth_y = mean_yc.item()
            face_msg.theta = mean_theta
            face_msg.a = mean_a.item()
            face_msg.b = mean_b.item()

            # Draw mouth ellipse
            cv2.circle(img, (int(round(mean_xc)), int(round(mean_yc))), radius, (130, 130, 130), thickness)
            # cv2.ellipse(img, (int(round(mean_xc)), int(round(mean_yc))), (int(round(mean_a)), int(round(mean_b))), int(round(mean_theta)), 0., 360, mouth_color)
            
            # Display Image Counter
            image_counter = image_counter + 1
            # if (image_counter % 11) == 10:
            #     rospy.loginfo("Images detected per second=%.2f", float(image_counter) / (time.time() - start_time))
        else:
            # Default face message
            face_msg.num_face = 0
            face_msg.mouth_open = False
            face_msg.mouth_x = 0
            face_msg.mouth_y = 0
            face_msg.theta = 0
            face_msg.a = 0
            face_msg.b = 0

            # Default mouth pose
            mouth_pose = None

        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_msg = bridge.cv2_to_imgmsg(im_rgb, encoding="rgb8")
        run.publish(im_msg, face_msg, mouth_pose)    
        
    return 0

if __name__ == '__main__':
    sys.exit(main())