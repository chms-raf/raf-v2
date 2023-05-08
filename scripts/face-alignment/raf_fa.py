#!/usr/bin/env python3
import rospy
import time
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import sys
import face_alignment
from std_msgs.msg import Bool, String
from skimage.measure import EllipseModel

class FKD(object):
    def __init__(self):
        # Params
        self.image = None
        self.br = CvBridge()
        self.raf_message = ""

        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(10)

        # Publishers
        self.pub = rospy.Publisher('face_detections', Image, queue_size=10)
        self.pub_msg = rospy.Publisher('mouth_open', Bool, queue_size=10)
        # self.result_pub = rospy.Publisher('arm_camera_results', Result, queue_size=10)

        # Subscribers
        rospy.Subscriber("/camera2/color/image_raw", Image, self.callback)
        rospy.Subscriber("/raf_message", String, self.message_callback)

    def callback(self, msg):
        self.image = self.convert_to_cv_image(msg)
        self._header = msg.header

    def message_callback(self, msg):
        self.raf_message = msg.data

    def get_img(self):
        result = self.image
        return result

    def get_message(self):
        result = self.raf_message
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

    def mouth_open_pdiff(self, Points, prev_dist, prev_mouth_open):
        # Check if mouth is open. This runs once per frame.
        # It takes the mouth points as an input
        # It then fits an ellipse to those points
        # If the ratio of the priciple axes a/b is > some threshold, the mouth is open
        # It returns the mouth state (open/closed)

        upper_lip = np.array([Points[50], Points[51],  Points[52]])
        lower_lip = np.array([Points[56], Points[57],  Points[58]])
        mean_upper_lip = np.mean(upper_lip, axis=0)
        mean_lower_lip = np.mean(lower_lip, axis=0)
        dist = np.linalg.norm(mean_upper_lip - mean_lower_lip)
        a = np.array([prev_dist, dist])

        percent_diff = ((prev_dist - dist) / np.mean(a)) * 100

        if percent_diff < -50:
            mouth_open = True
        elif percent_diff > 50:
            mouth_open = False
        else:
            mouth_open = prev_mouth_open

        return mouth_open, dist

    def publish(self, img, msg):
        self.pub.publish(img)
        self.pub_msg.publish(msg)
        self.loop_rate.sleep()

    def mouth_open_genForm(self, Points):
        mouth_points = Points[48:60]

        # print("Mouth Points: ", mouth_points)
        # print("\n")

        X = mouth_points[:,0]
        Y = mouth_points[:,1]

        X.shape = (12,1)
        Y.shape = (12,1)

        # print("X: ", X)
        # print("\n")
        # print("Y: ", Y)
        # print("\n")
        # print("----------------------------")

        A = np.hstack([X**2, X * Y, Y**2, X, Y])
        b = np.ones_like(X)


        x = np.linalg.lstsq(A, b)[0].squeeze()
        # print("Ellipse eqn: {0:.3}x^2 + {1:.3}xy + {2:.3}y^2 + {3:.3}x + {4:.3}y = 1".format(x[0], x[1], x[2], x[3], x[4]))

        x_coord = np.linspace(0, 640, 100)
        y_coord = np.linspace(0, 480, 100)
        X_coord, Y_coord = np.meshgrid(x_coord, y_coord)

        Z_coord = x[0]*X_coord**2 + x[1]*X_coord*Y_coord + x[2]*Y_coord**2 + x[3]*X_coord + x[4]*Y_coord

        print(Z_coord.shape)

        mouth_open = False

        return mouth_open

    def mouth_open_oneFrame(self, Points):
        mouth_points = Points[48:60]
        # X = mouth_points[:,0]
        # Y = mouth_points[:,1]
        # X.shape = (12,1)
        # Y.shape = (12,1)

        ell = EllipseModel()
        ell.estimate(mouth_points)

        xc, yc, a, b, theta = ell.params

        ratio = a / b

        # print("center = ",  (xc, yc))
        # print("angle of rotation = ",  theta)
        # print("axes = ", (a,b))
        # print(("Ratio = ", ratio))

        # print(("\n-------------------\n"))

        mouth_open = False

        return mouth_open, xc, yc, theta, a, b

    def mouth_open(self, Points):
        mouth_points = Points[48:60]
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

        if ratio > .5 and ratio < 2.0:
            mouth_open = True
        else:
            mouth_open = False

        return mouth_open, xc, yc, theta, a, b


def main():
    """ Face Detection """
    rospy.init_node("face_detection", anonymous=True)
    bridge = CvBridge()
    # start_time = time.time()
    image_counter = 0

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda', face_detector='sfd')

    run = FKD()
    numFace = None
    mouth_open = False
    mouth_open_msg = False
    mouth_timeout = 1.0
    movingAverageWindow = 1

    timer = None

    radius = 3
    thickness = -1

    xc_list = list()
    yc_list = list()
    theta_list = list()
    a_list = list()
    b_list = list()
    
    while not rospy.is_shutdown():
        # Get images
        img = run.get_img()

        if img is not None:

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

            if numFace is not None:

                image_counter = image_counter + 1

                msg = run.get_message()

                ready = msg == "Open your mouth when ready"

                Points = det.astype(int)

                mouth_open, xc, yc, theta, a, b = run.mouth_open(Points)

                xc_list.append(xc)
                yc_list.append(yc)
                theta_list.append(theta)
                a_list.append(a)
                b_list.append(b)
                if image_counter > movingAverageWindow:
                    xc_list.pop(0)
                    yc_list.pop(0)
                    theta_list.pop(0)
                    a_list.pop(0)
                    b_list.pop(0)

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

                # Only publish the mouth open trigger if we are ready to receive it
                if ready:
                    mouth_open_msg = mouth_open
                else:
                    mouth_open_msg = False

                if mouth_open:
                    mouth_color = (0, 255, 0)
                else:
                    mouth_color = (0, 0, 255)

                for i in range(60):
                    point = Points[i]

                    if i > 47:
                        # Mouth Color
                        color = mouth_color
                        radius = 3
                    else:
                        color = (255, 0, 0)
                        radius = 2

                    # cv2.circle(img, tuple(point), radius, color, thickness)

                    cv2.ellipse(img, (int(round(mean_xc)), int(round(mean_yc))), (int(round(mean_a)), int(round(mean_b))), int(round(mean_theta)), 0., 360, color)
                    # cv2.putText(img, str(i), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, .2, (0,255,255), 1, cv2.LINE_AA)

                # Display Image Counter
                # image_counter = image_counter + 1
                # if (image_counter % 11) == 10:
                    # rospy.loginfo("Images detected per second=%.2f", float(image_counter) / (time.time() - start_time))

            im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im_msg = bridge.cv2_to_imgmsg(im_rgb, encoding="rgb8")
            run.publish(im_msg, mouth_open_msg)    

        
        
    return 0

if __name__ == '__main__':
    sys.exit(main())