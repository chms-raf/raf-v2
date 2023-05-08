# from Xlib.protocol.rq import Bool
from numpy.lib.function_base import average
import rospy
import numpy as np
import os, cv2, random
import time
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import sys
import face_alignment
import collections
from std_msgs.msg import Bool, String

class FKD(object):
    def __init__(self):
        # Params
        self.image = None
        self.br = CvBridge()
        self.raf_message = ""

        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(100)

        # Publishers
        self.pub = rospy.Publisher('face_detections', Image, queue_size=10)
        self.pub_msg = rospy.Publisher('mouth_open', Bool, queue_size=10)
        # self.result_pub = rospy.Publisher('arm_camera_results', Result, queue_size=10)

        # Subscribers
        rospy.Subscriber("/camera1/color/image_raw", Image, self.callback)
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

    def publish(self, img, msg):
        self.pub.publish(img)
        self.pub_msg.publish(msg)
        self.loop_rate.sleep()


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
    prev_dist = 0
    mouth_open_msg = False

    timer = None

    radius = 3
    thickness = -1
    
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

                Points = det.astype(int)

                upper_lip = np.array([Points[50], Points[51],  Points[52]])
                lower_lip = np.array([Points[56], Points[57],  Points[58]])
                mean_upper_lip = np.mean(upper_lip, axis=0)
                mean_lower_lip = np.mean(lower_lip, axis=0)
                dist = np.linalg.norm(mean_upper_lip - mean_lower_lip)
                a = np.array([prev_dist, dist])

                msg = run.get_message()

                ready = msg == "Open your mouth when ready"

                image_counter = image_counter + 1
                # Create a slight buffer to prevent mouth from starting in open state
                if image_counter > 10:
                    percent_diff = ((prev_dist - dist) / np.mean(a)) * 100
                else:
                    percent_diff = 0

                # If mouth is open for 2 seconds it will automatically set the mouth to closed
                if timer is None:
                    if percent_diff < -50:
                        mouth_open = True
                        start_time = time.time()
                        timer = 0
                elif timer < 1.0:
                    if percent_diff > 50:
                        mouth_open = False
                        timer = None
                    else:
                        timer = time.time() - start_time
                else:
                    mouth_open = False
                    timer = None

                prev_dist = dist

                # Only publish the mouth open trigger if we are ready to receive it
                if ready:
                    mouth_open_msg = mouth_open
                else:
                    mouth_open_msg = False

                # print("Percent Difference: ", percent_diff)

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

                    cv2.circle(img, tuple(point), radius, color, thickness)
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