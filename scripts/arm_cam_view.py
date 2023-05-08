#!/usr/bin/env python3

# import some common libraries
import numpy as np
import cv2

import rospy, sys, math
from cv_bridge import CvBridge
from raf.msg import DetectionList, RafState
from sensor_msgs.msg import Image
from std_msgs.msg import String
from Xlib import display

class ArmCamView(object):
    def __init__(self):
        # Params
        self.image = None
        self.scene_image = None
        self.detections = DetectionList()
        self.br = CvBridge()
        self.raf_state = None
        self.viz_img = None

        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(15)

        # Publishers
        self.pub = rospy.Publisher('arm_camera_image', Image, queue_size=10)
        self.pub_cursor = rospy.Publisher('raf_cursor_angle', String, queue_size=10)
        self.pub_msg = rospy.Publisher('raf_message', String, queue_size=10)

        # Subscribers
        rospy.Subscriber("/raf_state", RafState, self.state_callback)
        rospy.Subscriber("/camera/color/image_raw", Image, self.callback)
        rospy.Subscriber("/scene_camera/color/image_raw", Image, self.scene_callback)
        rospy.Subscriber("/arm_camera_detections", DetectionList, self.detection_callback)

    def callback(self, msg):
        self.image = self.convert_to_cv_image(msg)
        self._header = msg.header

    def scene_callback(self, msg):
        self.scene_image = self.convert_to_cv_image(msg)

    def state_callback(self, msg):
        self.raf_state = msg
        
    def detection_callback(self, msg):
        self.detections = msg

    def init_colors(self):
        # Set up custom cv2 visualization parameters
        # Classes: [name, id]
        #               -
        #          [Plate,   0]
        #          [Bowl,  1]
        #          [Cup,  2]
        #          ...

        # Colors = [blue, green, red]
        color_plate = [255, 221, 51]
        color_bowl = [83, 50, 250]
        color_cup = [183, 209, 52]
        color_fork = [124, 0, 255]
        color_spoon = [55, 96, 255]
        color_knife = [51, 255, 221]
        color_pretzel = [83, 179, 36]
        color_carrot = [245, 61, 184]
        color_celery = [102, 255, 102]
        color_strawberry = [250, 183, 50]
        color_banana = [51, 204, 255]
        color_watermelon = [112, 224, 131]
        color_yogurt = [55, 250, 250]
        color_cottage_cheese = [179, 134, 89]
        color_beans = [240, 120, 140]
        color_gripper = [80, 80, 178]   
        self.colors = list([color_plate, color_bowl, color_cup, color_fork, color_spoon, color_knife, 
                    color_pretzel, color_carrot, color_celery, color_strawberry, color_banana, 
                    color_watermelon, color_yogurt, color_cottage_cheese, color_beans, color_gripper])

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
    
    def visualize_detections(self, det, img, colors):
        # Visualize using custom cv2 code
        det_cls = det.class_names
        det_clsId = det.class_ids
        det_scores = det.scores
        det_masks = det.masks

        # Create copies of the original image
        im = img.copy()
        output = img.copy()

        # Initialize lists
        masks = []
        masks_indices = []
        for i in range(len(det_clsId)):
            # Obtain current object mask as a numpy array (black and white mask of single object)
            current_mask = self.br.imgmsg_to_cv2(det_masks[i])

            # Find current mask indices
            mask_indices = np.where(current_mask==255)

            # Add to mask indices list
            if len(masks_indices) > len(det_clsId):
                masks_indices = []
            else:
                masks_indices.append(mask_indices)

            # Add to mask list
            if len(masks) > len(det_clsId):
                masks = []
            else:
                masks.append(current_mask)

        if len(masks) > 0:
            # Create composite mask
            composite_mask = sum(masks)

            # Clip composite mask between 0 and 255   
            composite_mask = composite_mask.clip(0, 255)

        # # Apply mask to image
        # masked_img = cv2.bitwise_and(im, im, mask=current_mask)

        # Find indices of object in mask
        # composite_mask_indices = np.where(composite_mask==255)

        for i in range(len(det_clsId)):
            # Select correct object color
            color = colors[det_clsId[i]]

            # Change the color of the current mask object
            im[masks_indices[i][0], masks_indices[i][1], :] = color

        # Apply alpha scaling to image to adjust opacity
        alpha = .4
        cv2.addWeighted(im, alpha, output, 1 - alpha, 0, output)

        for i in range(len(det_clsId)):
            # Draw Bounding boxes
            start_point = (det.boxes[i].x_offset, det.boxes[i].y_offset)
            end_point = (det.boxes[i].x_offset + det.boxes[i].width, det.boxes[i].y_offset + det.boxes[i].height)
            start_point2 = (det.boxes[i].x_offset + 2, det.boxes[i].y_offset + 2)
            end_point2 = (det.boxes[i].x_offset + det.boxes[i].width - 2, det.boxes[i].y_offset + 12)
            color = colors[det_clsId[i]]
            box_thickness =  1

            name = det_cls[i]
            score = det_scores[i]
            # conf = round(score.item() * 100, 1)
            conf = round(score * 100, 1)
            string = str(name) + ":" + str(conf) + "%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (det.boxes[i].x_offset + 2, det.boxes[i].y_offset + 10)
            fontScale = .3
            text_thickness = 1
            output = cv2.rectangle(output, start_point, end_point, color, box_thickness)
            output = cv2.rectangle(output, start_point2, end_point2, color, -1)     # Text box
            output = cv2.putText(output, string, org, font, fontScale, [0, 0, 0], text_thickness, cv2.LINE_AA, False)
            self.viz_img = output

        im_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        im_msg = self.br.cv2_to_imgmsg(im_rgb, encoding="rgb8")

        return im_msg
    
    def Selection(self):
        # Selection Parameters
        timer_running = False
        timer = 0
        timer_start = 0
        cursor_angle = 0
        dwell_time = 2         # Time to fill cursor and select item
        delay_time = .5        # Time before cursor begins to fill
        hl_id = None
        selected = False
        selected_item = None
        selected_item_cls = None

        raf_msg = "Select Desired Object"
        self.change_raf_message(raf_msg)

        while not selected and (self.raf_state.visualize_detections == 'selection' or self.raf_state.visualize_detections == 'normal+selection'):

            if self.detections is None:
                continue

            if self.raf_state.visualize_detections == 'selection' and self.image is None:
                continue

            if self.raf_state.visualize_detections == 'normal+selection' and self.viz_img is None:
                continue

            # Get Mouse Cursor Position
            data = display.Display().screen().root.query_pointer()._data

            # Create a copy of the original image
            if self.raf_state.visualize_detections == 'selection':
                output = self.image.copy()
            elif self.raf_state.visualize_detections == 'normal+selection':
                self.visualize_detections(self.detections, self.image, self.colors)
                output = self.viz_img.copy()

            item_ids = list(self.detections.class_ids)
            idx = [i for i, e in enumerate(item_ids) if e > 0 and e < 14 ]         # exclude gripper for now

            if len(idx) <= 0 or idx is None:
                continue

            for i in range(len(idx)):
                ul_x = self.detections.boxes[idx[i]].x_offset
                ul_y = self.detections.boxes[idx[i]].y_offset
                br_x = ul_x + self.detections.boxes[idx[i]].width
                br_y = ul_y + self.detections.boxes[idx[i]].height

                # GUI on desktop
                X1 = self.linear_map(0, 1280, 321, 1601, ul_x)
                X2 = self.linear_map(0, 1280, 321, 1601, br_x)
                Y1 = self.linear_map(0, 720, 165, 884, ul_y)
                Y2 = self.linear_map(0, 720, 165, 884, br_y)

                # print("(", data["root_x"], ", ", data["root_y"], ")")
                # print("{:10}, {:10}, {:10}, {:10}".format(X1, X2, Y1, Y1))
                # print("------------------------------------")

                # Check if cursor is inside the bounding box
                # TODO: need to also check if cursor has been stable for a length of time
                if data["root_x"] > X1 and data["root_x"] < X2 and data["root_y"] > Y1 and data["root_y"] < Y2:
                    color = [0, 220, 255]
                    thickness = 2

                    # return id of highlighted item
                    hl_id = i

                    # Cursor enters box
                    if not timer_running and hl_id == i:
                        timer_start = rospy.get_time()
                        timer_running = True
                        cursor_angle = self.linear_map(delay_time, dwell_time, 0, 2*math.pi, np.clip(timer, delay_time, dwell_time))
                        self.pub_cursor.publish(str(cursor_angle))

                    # Cursor remains in box
                    if timer_running and hl_id == i:
                        timer = rospy.get_time() - timer_start
                        cursor_angle = self.linear_map(delay_time, dwell_time, 0, 2*math.pi, np.clip(timer, delay_time, dwell_time))
                        self.pub_cursor.publish(str(cursor_angle))

                else:
                    color = [0,0,0]
                    thickness = 1

                    # Cursor leaves box
                    if timer_running and hl_id == i:
                        timer_running = False
                        timer_start = 0
                        timer = 0
                        cursor_angle = self.linear_map(delay_time, dwell_time, 0, 2*math.pi, np.clip(timer, delay_time, dwell_time))
                        self.pub_cursor.publish(str(cursor_angle))

                if timer > dwell_time + delay_time:
                    # print("Object Selected!")
                    timer_running = False
                    timer_start = 0
                    timer = 0
                    cursor_angle = self.linear_map(delay_time, dwell_time, 0, 2*math.pi, np.clip(timer, delay_time, dwell_time))
                    self.pub_cursor.publish(str(cursor_angle))

                    selected = True
                    selected_item = i
                    selected_item_cls = self.detections.class_names[selected_item]
                    color = [0, 255, 0]
                    thickness = 2

                # Draw Bounding boxes
                start_point = (ul_x, ul_y)
                end_point = (br_x, br_y)
                output = cv2.rectangle(output, start_point, end_point, color, thickness)

            im_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            im_msg = self.br.cv2_to_imgmsg(im_rgb, encoding="rgb8")
            self.publish(im_msg)

        raf_msg = "Object Selected: " + str(selected_item_cls)
        self.change_raf_message(raf_msg)
        rospy.sleep(1.0)

        return selected_item, selected_item_cls
    
    def linear_map(self, x1, x2, y1, y2, x):
        slope = 1.0 * (y2 - y1) / (x2 - x1)
        y = (slope * (x - x1)) +  y1
        return y
    
    def change_raf_message(self, msg):
        self.RAF_msg = msg
        self.pub_msg.publish(self.RAF_msg)

    def publish(self, img):
        self.pub.publish(img)
        self.loop_rate.sleep()


def main():
    """ Mask RCNN Object Detection with Detectron2 """
    rospy.init_node("arm_cam_view", anonymous=True)
    bridge = CvBridge()

    run = ArmCamView()
    run.init_colors()
    while not rospy.is_shutdown():
        # Get images
        img = run.image

        if run.raf_state is None:
            continue

        if not run.raf_state.enable_arm_detections:
            if img is not None:
                im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                im_msg = bridge.cv2_to_imgmsg(im_rgb, encoding="rgb8")
                run.publish(im_msg)
        
        else:
            # Get detections
            det = run.detections

            if det is None:
                continue

            # Visualize based on RAF state
            if run.raf_state.visualize_detections == "normal":
                if img is not None:
                    viz_msg = run.visualize_detections(det, img, run.colors)
                    run.publish(viz_msg)
            elif run.raf_state.visualize_detections == "selection" or run.raf_state.visualize_detections == "normal+selection":
                if img is not None:
                    selected, item = run.Selection()
                    # print("Selected Item: ", item)
            else:
                if img is not None:
                    im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    im_msg = bridge.cv2_to_imgmsg(im_rgb, encoding="rgb8")
                    run.publish(im_msg)

    return 0

if __name__ == '__main__':
    sys.exit(main())