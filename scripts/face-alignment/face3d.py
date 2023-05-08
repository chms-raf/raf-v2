import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
import sys
import face_alignment
import time
from skimage.measure import EllipseModel
from tf import TransformBroadcaster, transformations
import pyrealsense2
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from collections import deque

# print(face_alignment.__file__)

class FKD(object):
    def __init__(self):
        # Params
        self.image = None
        self.br = CvBridge()
        self.det_type = "2D"
        self.orientation = False

        # Camera params
        self.intrinsics = pyrealsense2.intrinsics()
        self.camera_frame = "camera_color_frame"

        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(100)

        # Publishers
        self.pub = rospy.Publisher('face_detections', Image, queue_size=10)
        # self.result_pub = rospy.Publisher('arm_camera_results', Result, queue_size=10)
        self.tf_pub = TransformBroadcaster()

        # Subscribers
        rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.info_callback)
        # rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        rospy.Subscriber("/camera/depth_registered/sw_registered/image_rect", Image, self.depth_callback)

    def image_callback(self, msg):
        self.image = self.convert_to_cv_image(msg)

    def info_callback(self, msg):
        self.intrinsics.width = msg.width
        self.intrinsics.height = msg.height
        self.intrinsics.ppx = msg.K[2]
        self.intrinsics.ppy = msg.K[5]
        self.intrinsics.fx = msg.K[0]
        self.intrinsics.fy = msg.K[4]
        # self.intrinsics.model = msg.distortion_model
        self.intrinsics.model = pyrealsense2.distortion.none
        self.intrinsics.coeffs = [i for i in msg.D]

    def depth_callback(self, msg):
        self.depth_array = self.convert_depth_image(msg)
        self._header = msg.header

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

    def convert_depth_image(self, image_msg):
        if image_msg is not None:
            depth_image = self.br.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
            depth_array = np.array(depth_image, dtype=np.float32)
            return depth_array
        else:
            return None

    def pixel_to_world(self, x, y, z):
        result = pyrealsense2.rs2_deproject_pixel_to_point(self.intrinsics, [x, y], z)
        #result[0]: right, result[1]: down, result[2]: forward
        # return result[2], -result[0], -result[1]
        return result[0], result[1], result[2]

    def compute_face_depth(self, landmarks):        
        # Average depth of nostrils, eyes, and lips seems to work pretty well
        points = landmarks[31:60, 0:2]
        
        depth_list = []
        for i in range(len(points)):
            depth = int(self.depth_array[int(points[i, 1]), int(points[i, 0])])
            if depth != 0:
                depth_list.append(depth)
            else:
                pass
                # print("Zero depth for coordinate: ", [int(points[ii, 1]), int(points[ii, 0])])
                
        depth_avg = np.mean(depth_list)
        depth_avg = depth_avg / 1000

        # print("Number of landmarks to test: ", len(points))
        # print("Number of landmarks tested: ", len(depth_list))
        # print("Average Depth: ", depth_avg)
        # print("-------------------------------\n")

        return depth_avg

    def compute_mouth_3D(self, landmarks):        
        # Average depth of nostrils, eyes, and lips seems to work pretty well
        xp = landmarks[48:60, 0]
        yp = landmarks[48:60, 1]

        x = []
        y = []
        z = []
        for i in range(len(xp)):
            depth = int(self.depth_array[int(yp[i]), int(xp[i])])
            depth = depth / 1000

            xw, yw, zw = self.pixel_to_world(xp[i], yp[i], depth)
            x.append(xw)
            y.append(yw)
            z.append(zw)
            # if depth == 0:
            #     print("Zero depth for coordinate: ", [int(x[i]), int(y[i])])

        # mouth_points = np.transpose(np.array([x, y, z]))
        # print(mouth_points)
        # print("---------------------------\n")

        return x, y, z

    def ls_ellipsoid(self,xx,yy,zz):                                  
        #finds best fit ellipsoid. Found at http://www.juddzone.com/ALGORITHMS/least_squares_3D_ellipsoid.html
        #least squares fit to a 3D-ellipsoid
        #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz  = 1
        #
        # Note that sometimes it is expressed as a solution to
        #  Ax^2 + By^2 + Cz^2 + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz  = 1
        # where the last six terms have a factor of 2 in them
        # This is in anticipation of forming a matrix with the polynomial coefficients.
        # Those terms with factors of 2 are all off diagonal elements.  These contribute
        # two terms when multiplied out (symmetric) so would need to be divided by two
        
        # change xx from vector of length N to Nx1 matrix so we can use hstack
        x = xx[:,np.newaxis]
        y = yy[:,np.newaxis]
        z = zz[:,np.newaxis]
        
        #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz = 1
        J = np.hstack((x*x,y*y,z*z,x*y,x*z,y*z, x, y, z))
        K = np.ones_like(x) #column of ones
        
        #np.hstack performs a loop over all samples and creates
        #a row in J for each x,y,z sample:
        # J[ix,0] = x[ix]*x[ix]
        # J[ix,1] = y[ix]*y[ix]
        # etc.
        
        JT=J.transpose()
        JTJ = np.dot(JT,J)
        InvJTJ=np.linalg.inv(JTJ)
        ABC= np.dot(InvJTJ, np.dot(JT,K))

        # Rearrange, move the 1 to the other side
        #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz - 1 = 0
        #    or
        #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz + J = 0
        #  where J = -1
        eansa=np.append(ABC,-1)

        return (eansa)

    def polyToParams3D(self,vec,printMe):                             
        #gets 3D parameters of an ellipsoid. Found at http://www.juddzone.com/ALGORITHMS/least_squares_3D_ellipsoid.html
        # convert the polynomial form of the 3D-ellipsoid to parameters
        # center, axes, and transformation matrix
        # vec is the vector whose elements are the polynomial
        # coefficients A..J
        # returns (center, axes, rotation matrix)
        
        #Algebraic form: X.T * Amat * X --> polynomial form
        
        if printMe: print('\npolynomial\n',vec)
        
        Amat=np.array(
        [
        [ vec[0],     vec[3]/2.0, vec[4]/2.0, vec[6]/2.0 ],
        [ vec[3]/2.0, vec[1],     vec[5]/2.0, vec[7]/2.0 ],
        [ vec[4]/2.0, vec[5]/2.0, vec[2],     vec[8]/2.0 ],
        [ vec[6]/2.0, vec[7]/2.0, vec[8]/2.0, vec[9]     ]
        ])
        
        if printMe: print('\nAlgebraic form of polynomial\n',Amat)
        
        #See B.Bartoni, Preprint SMU-HEP-10-14 Multi-dimensional Ellipsoidal Fitting
        # equation 20 for the following method for finding the center
        A3=Amat[0:3,0:3]
        A3inv=np.linalg.inv(A3)
        ofs=vec[6:9]/2.0
        center=-np.dot(A3inv,ofs)
        if printMe: print('\nCenter at:',center)
        
        # Center the ellipsoid at the origin
        Tofs=np.eye(4)
        Tofs[3,0:3]=center
        R = np.dot(Tofs,np.dot(Amat,Tofs.T))
        if printMe: print('\nAlgebraic form translated to center\n',R,'\n')
        
        R3=R[0:3,0:3]
        R3test=R3/R3[0,0]
        # print('normed \n',R3test)
        s1=-R[3, 3]
        R3S=R3/s1
        (el,ec)=np.linalg.eig(R3S)
        
        recip=1.0/np.abs(el)
        axes=np.sqrt(recip)
        if printMe: print('\nAxes are\n',axes  ,'\n')
        
        inve=np.linalg.inv(ec) #inverse is actually the transpose here
        if printMe: print('\nRotation matrix\n',inve)
        return (center,axes,inve)

    def rot2eul(self, R):
        beta = -np.arcsin(R[2,0])
        alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
        gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
        return np.array((alpha, beta, gamma))

    def compute_ellipse_3D(self, x, y, z):
        #get convex hull
        surface = np.stack((x, y, z), axis=-1)
        hullV = ConvexHull(surface)
        lH = len(hullV.vertices)
        hull = np.zeros((lH,3))
        for i in range(len(hullV.vertices)):
            hull[i] = surface[hullV.vertices[i]]
        
        hull = np.transpose(hull)         
                    
        #fit ellipsoid on convex hull
        eansa = self.ls_ellipsoid(hull[0], hull[1], hull[2]) #get ellipsoid polynomial coefficients
        # print("coefficients:"  , eansa)
        center, axes, inve = self.polyToParams3D(eansa, False)   #get ellipsoid 3D parameters
        # print("center:"        , center)
        # print("axes:"          , axes)
        # print("rotationMatrix:", inve)
        # print("---------------------------\n")

        angles = self.rot2eul(inve)

        return angles

    def publish(self, img, mouth_pose, mouth_open):
        self.pub.publish(img)
        self.loop_rate.sleep()

        if mouth_pose is not None:
            if self.camera_frame == "camera_color_frame":
                self.tf_pub.sendTransform((mouth_pose[0], mouth_pose[1], mouth_pose[2]),
                                    transformations.quaternion_from_euler(0, 0, 0),
                                    rospy.Time.now(),
                                    "mouth",
                                    self.camera_frame)
                if mouth_open is not None and not mouth_open:
                    self.tf_pub.sendTransform((0, -0.06, -0.5),
                                        (0, 0, 0, 1),
                                        rospy.Time.now(),
                                        "desired_cam_frame",
                                        "mouth")
                # self.tf_pub.sendTransform((mouth_pose[2], -mouth_pose[0], -mouth_pose[1]),
                #                     transformations.quaternion_from_euler(0, 0, 0),
                #                     rospy.Time.now(),
                #                     "mouth",
                #                     self.camera_frame)
            elif self.camera_frame == "camera_color_optical_frame":
                self.tf_pub.sendTransform((mouth_pose[0], mouth_pose[1], mouth_pose[2]),
                                    transformations.quaternion_from_euler(mouth_pose[3], mouth_pose[4], mouth_pose[5]),
                                    rospy.Time.now(),
                                    "mouth",
                                    self.camera_frame)

            else:
                rospy.logwarn("Error broadcasting mouth tf.")
                pass

    def mouth_open(self, Points):

        try:
            mouth_points = Points[48:60, 0:2]
            # X = mouth_points[:,0]
            # Y = mouth_points[:,1]
            # X.shape = (12,1)
            # Y.shape = (12,1)

            ell = EllipseModel()
            ell.estimate(mouth_points)

            xc, yc, a, b, theta = ell.params

            temp = [a, b]
            a = max(temp)
            b = min(temp)

            ratio = a / b

            # print("center = ",  (xc, yc))
            # print("angle of rotation = ",  theta)
            # print("axes = ", (a,b))
            # print(("Ratio = ", ratio))

            # print(("\n-------------------\n"))

            if ratio > .7 and ratio < 2.0:
                mouth_open = True
            else:
                mouth_open = False
        except:
            rospy.logwarn("Mouth ellipse error.")
            # TODO: if error, just draw ellipse from previous frame, or don't return one at all
            mouth_open = False
            xc, yc, theta, a, b = 0, 0, 0, 0, 0
        

        return mouth_open, xc, yc, theta, a, b

    def compute_head_pose(self, Points, img):
        Points = Points[:, 0:2]
        # Points order: nose tip, chin, left eye left corner, right eye right corner, left mouth left corner, right mouth right corner
        # Left and right are from the image perspective, not the face
        image_points = np.array([Points[29], Points[8], Points[36], Points[45], Points[48], Points[54]], dtype="double")
        model_points = np.array([(0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)], dtype="double")
        camera_matrix = np.array([[self.intrinsics.fx, 0, self.intrinsics.ppx],[0, self.intrinsics.fy, self.intrinsics.ppy],[0, 0, 1]], dtype="double")
        dist_coeffs = np.array([self.intrinsics.coeffs])
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        p1 = ( int(image_points[0][0]), int(image_points[0][1]))
        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        try:
            img = cv2.line(img, p1, p2, (255,0,0), 2)
        except:
            pass

        RotMatrix = cv2.Rodrigues(rotation_vector)[0]
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(RotMatrix)

        return img, angles


def main():
    """ Face Detection """
    rospy.init_node("face_pbvs", anonymous=True)
    bridge = CvBridge()

    start_time = time.time()

    mouthX = 0
    mouthY = 0
    mouthZ = 0

    run = FKD()

    # TODO: modify code shapes and slices to work with both 2D and 3D
    if run.det_type == "3D":
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda', face_detector='sfd')
    elif run.det_type == "2D":
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda', face_detector='sfd')
    else:
        rospy.logerr("Invalid Detection Type.")
    
    movingAverageWindow = 1
    image_counter = 0

    radius = 3
    thickness = -1

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
    
    while not rospy.is_shutdown():
        # Get images
        img = run.get_img()

        if img is None:
            rospy.logwarn("No image detected.")
            continue

        try:
            det = fa.get_landmarks(img)[-1]
            # print(det.shape)
        except:
            rospy.logwarn_once("No face detected.")
            # Even if no faces detected, still want to publish an image
            im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im_msg = bridge.cv2_to_imgmsg(im_rgb, encoding="rgb8")
            run.publish(im_msg, None, None)
            continue
        
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

        image_counter = image_counter + 1
        Points = det.astype(int)

        mouth_open, xc, yc, theta, a, b = run.mouth_open(Points)


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

        mouth_points = np.array(Points[48:60, :])
        mouth_mean = np.mean(mouth_points, 0)

        mouthX_pix = mouth_mean[0]
        mouthY_pix = mouth_mean[1]
        # mouthZ_pix = run.compute_face_depth(Points)
        mouthZ_pix = 0.5

        if mouthX_pix is not None:
            mouthX, mouthY, mouthZ = run.pixel_to_world(mouthX_pix, mouthY_pix, mouthZ_pix)
        else:
            mouthX = 0
            mouthY = 0
            mouthZ = 0

        # Compute head orientation and set the mouth pose vector
        if run.orientation:
            try:
                # img, angles = run.compute_head_pose(Points, img)
                x, y, z = run.compute_mouth_3D(Points)
                angles = run.compute_ellipse_3D(x, y, z)
            except:
                mouth_pose = [mouthX, mouthY, mouthZ, 0, 0, 0]
            else:
                mouth_pose = [mouthX, mouthY, mouthZ, angles[0], angles[1], angles[2]]
        else:
            mouth_pose = [mouthX, mouthY, mouthZ, 0, 0, 0]

        # print(mouth_pose)
        # print("-------------------------------\n")

        # print("Mouth Mean: ", [mouthX, mouthY, mouthZ])
        # print("---------------------------------------\n")

        if mouth_open:
            mouth_color = (0, 255, 0)
        else:
            mouth_color = (0, 0, 255)

        # Draw facial keypoints
        # for i in range(60):
        #     point = Points[i][0:2]

        #     if i > 47:
        #         # Mouth Color
        #         color = mouth_color
        #         radius = 3
        #     else:
        #         color = (255, 0, 0)
        #         radius = 2

        #     cv2.circle(img, tuple(point), radius, color, thickness)
        #     cv2.putText(img, str(i), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, .2, (0,255,255), 1, cv2.LINE_AA)

        # Draw mount ellipse and center
        # cv2.circle(img, tuple(mouth_mean[0:2].astype(int)), 5, (0, 220, 255), thickness)
        cv2.ellipse(img, (int(round(mean_xc)), int(round(mean_yc))), (int(round(mean_a)), int(round(mean_b))), int(round(mean_theta)), 0., 360, mouth_color)
        
        # Display Image Counter
        if (image_counter % 11) == 10:
            rospy.loginfo("Images detected per second=%.2f", float(image_counter) / (time.time() - start_time))

        ### FPS Measurments ###
        # Stereo depth camera (USB 3.0), no rviz, no FKD            - 95 FPS
        # Stereo depth camera (USB 3.0), with rviz, no FKD          - 93 FPS
        # Stereo depth camera (USB 3.0), no rviz, with FKD (2D)     - 14.8 FPS
        # Stereo depth camera (USB 3.0), with rviz, with FKD (2D)   - 14 FPS
        # Stereo depth camera (USB 3.0), no rviz, with FKD (3D)     - 11 FPS
        # Gen3 camera to CSU network, launch on laptop, subscribe on desktop - say 75 FPS but looks more like 10 with a huge delay

        # Convert image to message to publish
        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_msg = bridge.cv2_to_imgmsg(im_rgb, encoding="rgb8")
        run.publish(im_msg, mouth_pose, mouth_open)

    return 0

if __name__ == '__main__':
    sys.exit(main())