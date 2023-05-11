#!/usr/bin/env python3

# This code is responsible for handling which camera stream to display on the GUI.
# You can switch between the scene camera and the arm camera as well as control different
# visualiztion options.

# Author: Jack Schultz
# Created 1/26/2023

# import some common libraries
import numpy as np
import cv2

import rospy, sys, math
from cv_bridge import CvBridge
from raf.msg import DetectionList, RafState, Selection, FaceDetection
from sensor_msgs.msg import Image, RegionOfInterest
from geometry_msgs.msg import Point
from std_msgs.msg import String
from Xlib import display
import tf2_ros, tf

class CamView(object):
    def __init__(self):
        # Params
        self.arm_image = None
        self.scene_image = None
        self.face_detection = FaceDetection()
        self.img = None
        self.arm_detections = DetectionList()
        self.scene_detections = DetectionList()
        self.detections = None
        self.br = CvBridge()
        self.raf_state = None
        self.viz_img = None
        self.select_img = None
        self.dwell_time = 1.5

        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(15)

        # Publishers
        self.pub = rospy.Publisher('arm_camera_image', Image, queue_size=10)
        self.pub_cursor = rospy.Publisher('raf_cursor_angle', String, queue_size=10)
        self.pub_msg = rospy.Publisher('raf_message', String, queue_size=10)
        self.request_pub = rospy.Publisher('raf_state_request', String, queue_size=10)
        self.action_pub = rospy.Publisher('raf_action', String, queue_size=10)
        self.selection_pub = rospy.Publisher('raf_selection', Selection, queue_size=10)

        # Subscribers
        rospy.Subscriber("/raf_state", RafState, self.state_callback)
        rospy.Subscriber("/camera/color/image_raw", Image, self.arm_callback)
        rospy.Subscriber("/scene_camera/color/image_raw", Image, self.scene_callback)
        rospy.Subscriber("/arm_camera_detections", DetectionList, self.arm_detection_callback)
        rospy.Subscriber("/scene_camera_detections", DetectionList, self.scene_detection_callback)
        rospy.Subscriber("/face_detection", FaceDetection, self.face_detection_callback)

        self.tfBuffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tfBuffer)

    def arm_callback(self, msg):
        self.arm_image = self.convert_to_cv_image(msg)
        self._header = msg.header

    def scene_callback(self, msg):
        self.scene_image = self.convert_to_cv_image(msg)

    def face_detection_callback(self, msg):
        self.face_detection = msg

    def state_callback(self, msg):
        self.raf_state = msg
        
    def arm_detection_callback(self, msg):
        # temp = msg

        temp = self.sort_detections(msg)
        self.arm_detections = DetectionList()
        for i in range(len(temp)):
            self.arm_detections.class_ids.append(temp[i][0])
            self.arm_detections.class_names.append(temp[i][1])
            self.arm_detections.scores.append(temp[i][2])
            self.arm_detections.boxes.append(temp[i][3])
            self.arm_detections.masks.append(temp[i][4])

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

    def scene_detection_callback(self, msg):
        self.scene_detections = msg

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
            org2 = (det.boxes[i].x_offset + 2, det.boxes[i].y_offset + 30)
            fontScale = .3
            text_thickness = 1
            output = cv2.rectangle(output, start_point, end_point, color, box_thickness)
            output = cv2.rectangle(output, start_point2, end_point2, color, -1)     # Text box
            output = cv2.putText(output, string, org, font, fontScale, [0, 0, 0], text_thickness, cv2.LINE_AA, False)
            # output = cv2.putText(output, str(i), org2, font, fontScale, [255, 0, 0], text_thickness, cv2.LINE_AA, False)
            self.viz_img = output

        im_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        im_msg = self.br.cv2_to_imgmsg(im_rgb, encoding="rgb8")

        return im_msg, im_rgb
    
    def Selection(self):
        # Selection Parameters
        self.bridge = CvBridge()
        timer_running = False
        timer = 0
        timer_start = 0
        cursor_angle = 0
        dwell_time = self.dwell_time         # Time to fill cursor and select item
        delay_time = .5        # Time before cursor begins to fill
        hl_id = None
        selected = False
        selected_item = None
        selected_item_cls = None
        selection_msg = Selection()

        raf_msg = "Select Desired Object"
        self.change_raf_message(raf_msg)

        while not selected and self.raf_state.system_state == 'selection':

            if self.raf_state.view == "arm":
                self.img = self.arm_image
            elif self.raf_state.view == 'scene':
                self.img = self.scene_image

            if self.img is None:
                continue

            if self.raf_state.enable_arm_detections:
                self.detections = self.arm_detections
            elif self.raf_state.enable_scene_detections:
                self.detections = self.scene_detections

            if self.detections is None:
                continue

            # Get Mouse Cursor Position
            data = display.Display().screen().root.query_pointer()._data

            # Create a copy of the original image
            if self.raf_state.visualize_detections == 'selection':
                output = self.img.copy()
            elif self.raf_state.visualize_detections == 'normal+selection':
                self.visualize_detections(self.detections, self.img, self.colors)
                output = self.viz_img.copy()

            item_ids = list(self.detections.class_ids)
            idx = [i for i, e in enumerate(item_ids) if e > 0 and e < 14 ]         # exclude plate and gripper for now

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
                    selected_item_cls = self.detections.class_names[idx[selected_item]]
                    color = [0, 255, 0]
                    thickness = 2

                    # Build selection message
                    selection_msg.box = self.detections.boxes[idx[selected_item]]
                    selection_msg.class_id = self.detections.class_ids[idx[selected_item]]
                    selection_msg.class_name = selected_item_cls
                    selection_msg.score = self.detections.scores[idx[selected_item]]
                    selection_msg.centroid, gp, c = self.compute_centroid(self.detections.masks[idx[selected_item]])
                    selection_msg.transform = self.get_tf('base_link', 'camera_link')

                    # Draw centroid on image for testing
                    # centX = selection_msg.centroid.x
                    # centY = selection_msg.centroid.y
                    output = cv2.circle(output, (int(c[0]), int(c[1])), 3, [255, 0, 0], -1)
                    rospy.sleep(0.5)

                # Draw Bounding boxes
                start_point = (ul_x, ul_y)
                end_point = (br_x, br_y)
                output = cv2.rectangle(output, start_point, end_point, color, thickness)
                self.select_img = output

            im_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            im_msg = self.br.cv2_to_imgmsg(im_rgb, encoding="rgb8")
            self.publish(im_msg, im_rgb)

        raf_msg = "Object Selected: " + str(selected_item_cls)
        self.change_raf_message(raf_msg)
        rospy.sleep(0.5)

        return selected_item_cls, (data["root_x"], data["root_y"]), selection_msg
    
    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype = "float32")

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

        rotrect = cv2.minAreaRect(cntrs[0])
        box = cv2.boxPoints(rotrect)
        box = np.int0(box)

        # Reorder the box points
        box = self.order_points(box)
        box = np.int0(box)

        # Compute important points
        c = np.mean(box, axis=0)
        p = Point()
        p.x = c[0]
        p.y = c[1]
        p.z = 0.0

        m = np.mean(box[2:], axis=0)
        gp = np.squeeze(np.mean(np.array([[m],[c]]), axis=0))
        return p, gp, c
    
    def get_tf(self, parent, child):
        try:
            trans = self.tfBuffer.lookup_transform(parent, child, rospy.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            trans = None
        return trans
    
    def actionSelection(self, item_cls, mouse_pos):
        # Selection Parameters
        selected = False
        timer_running = False
        timer = 0
        timer_start = 0
        cursor_angle = 0
        dwell_time = self.dwell_time         # Time to fill cursor and select item
        delay_time = .5        # Time before cursor begins to fill
        hl_id = None
        selected_action = None

        actions = self.actionLookup(item_cls)

        while not selected and self.raf_state.system_state == 'selection':
            if self.raf_state.view == "arm":
                self.img = self.arm_image
            elif self.raf_state.view == 'scene':
                self.img = self.scene_image

            if self.img is None:
                continue

            # Create a copy of the original image
            if self.raf_state.visualize_detections == 'selection':
                output = self.img.copy()
            elif self.raf_state.visualize_detections == 'normal+selection':
                self.visualize_detections(self.detections, self.img, self.colors)
                output = self.viz_img.copy()

            img, actionBoxes = self.drawActionMenu(output, actions, mouse_pos)

            # Get Mouse Cursor Position
            data = display.Display().screen().root.query_pointer()._data
            mx = self.linear_map(321, 1601, 0, 1280, data["root_x"])
            my = self.linear_map(165, 884, 0, 720, data["root_y"])

            # Insert selction code here!!!
            for i in range(len(actionBoxes)):

                # Check if cursor is inside the bounding box
                # TODO: need to also check if cursor has been stable for a length of time
                if mx > actionBoxes[i].x_offset and mx < actionBoxes[i].x_offset + actionBoxes[i].width and my > actionBoxes[i].y_offset and my < actionBoxes[i].y_offset + actionBoxes[i].height:
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
                    selected_action = actions[selected_item]
                    color = [0, 255, 0]
                    thickness = 2

                # Draw Bounding boxes
                start_point = (int(actionBoxes[i].x_offset), int(actionBoxes[i].y_offset))
                end_point = (int(actionBoxes[i].x_offset + actionBoxes[i].width), int(actionBoxes[i].y_offset + actionBoxes[i].height))
                output = cv2.rectangle(output, start_point, end_point, color, thickness)
                # self.actionSelect_img = output

            im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im_msg = self.br.cv2_to_imgmsg(im_rgb, encoding="rgb8")
            self.publish(im_msg, im_rgb)

        raf_msg = "Action Selected: " + str(selected_action)
        self.change_raf_message(raf_msg)
        rospy.sleep(1.0)

        return selected_action
    
    def actionLookup(self, item_cls):

        if item_cls is None:
            actions = None
        else:
            # TODO: Auto-create this list on initialization based on the detection model
            action_table = {'Plate': [], 'Bowl': [], 'Cup': ['Pick', 'Place', 'Sip', 'Drink'], 
                            'Fork': ['Pick', 'Place'], 'Spoon': ['Pick', 'Place'], 
                            'Knife': ['Pick', 'Place'], 'Pretzel': ['Grasp', 'Skewer', 'Scoop'], 
                            'Carrot': ['Grasp', 'Skewer', 'Scoop'], 'Celery': ['Grasp', 'Skewer', 'Scoop'], 
                            'Strawberry': ['Grasp', 'Skewer', 'Scoop'], 'Banana': ['Grasp', 'Skewer', 'Scoop'], 
                            'Watermelon': ['Grasp', 'Skewer', 'Scoop'], 'Yogurt': ['Grasp', 'Skewer', 'Scoop'], 
                            'Cottage_cheese': ['Grasp', 'Skewer', 'Scoop'], 'Beans': ['Grasp', 'Skewer', 'Scoop'], 
                            'Gripper': [] }
            
            actions = action_table[item_cls]

            # Add Cancel option to list of actions
            actions.append('Cancel')

        return actions
    
    def drawActionMenu(self, img, actions, mouse_pos):
        r = 100
        n = len(actions)
        theta = np.linspace(np.pi, np.pi*2, n-1)

        mx = self.linear_map(321, 1601, 0, 1280, mouse_pos[0])
        my = self.linear_map(165, 884, 0, 720, mouse_pos[1])

        fontScale = 0.7
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        padx = 5
        pady = 5

        xc = list()
        yc = list()
        actionBoxes = list()
        for i in range(n):

            if i < n-1:
                # Actions in semi-circle above mouse cursor
                xc.append(mx + (r*np.cos(theta[i])))
                yc.append(my + (r*np.sin(theta[i])))
            else:
                # Cancel directly below mouse cursor
                xc.append(mx)
                yc.append(my + r)
                
            # Draw circle at center spot
            img = cv2.circle(img, (int(xc[i]), int(yc[i])), 5, [0, 0, 255], 2)

            # Draw text
            text = actions[i]
            textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]

            textWidth = textsize[0]
            textHeight = textsize[1]

            rect = RegionOfInterest()
            rect.x_offset = xc[i] - textWidth/2 - padx; rect.y_offset = yc[i] - textHeight/2 - pady
            rect.width = textWidth + 2*padx
            rect.height = textHeight + 2*pady

            actionBoxes.append(rect)

            img = cv2.rectangle(img, (int(rect.x_offset), int(rect.y_offset)), (int(rect.x_offset + rect.width), int(rect.y_offset + rect.height)), [255, 255, 255], -1)
            img = cv2.putText(img, text, (int(xc[i] - textWidth/2), int(yc[i] + textHeight/2)), font, fontScale, [0, 0, 0], thickness)

        return img, actionBoxes

    def linear_map(self, x1, x2, y1, y2, x):
        slope = 1.0 * (y2 - y1) / (x2 - x1)
        y = (slope * (x - x1)) +  y1
        return y
    
    def change_raf_message(self, msg):
        self.RAF_msg = msg
        self.pub_msg.publish(self.RAF_msg)

    def visualize_face_detections(self, img):
        det = self.face_detection

        if det.mouth_open:
            mouth_color = (0, 255, 0)
        else:
            mouth_color = (255, 0, 0)

        # Draw mouth ellipse
        radius = 3
        thickness = -1
        cv2.circle(img, (int(round(det.mouth_x)), int(round(det.mouth_y))), radius, (0, 0, 255), thickness)
        cv2.ellipse(img, (int(round(det.mouth_x)), int(round(det.mouth_y))), (int(round(det.a)), int(round(det.b))), int(round(det.theta)), 0., 360, mouth_color)
        
        im_msg = self.br.cv2_to_imgmsg(img, encoding="rgb8")

        return im_msg

    def publish(self, img_msg, img_rgb):
        if self.raf_state.visualize_face_detections:
            img_msg = self.visualize_face_detections(img_rgb)
        self.pub.publish(img_msg)
        self.loop_rate.sleep()

def main():
    """ Mask RCNN Object Detection with Detectron2 """
    rospy.init_node("arm_cam_view", anonymous=True)
    bridge = CvBridge()

    run = CamView()
    run.init_colors()
    while not rospy.is_shutdown():

        if run.raf_state is None:
            continue

        # Check which view is selected
        if run.raf_state.view == "arm":
            run.img = run.arm_image
        elif run.raf_state.view == 'scene':
            run.img = run.scene_image
        else:
            run.img = run.arm_image

        if run.img is None:
            continue

        if run.raf_state.system_state == 'selection':
            # Check if detections are enabled
            # TODO: In the current setup, arm and scene detections are not allowed at the same time
            if run.raf_state.enable_arm_detections:
                run.detections = run.arm_detections

                if run.detections is None:
                    continue

                # Visualize based on RAF state
                if run.raf_state.visualize_detections == "normal":
                    viz_msg, viz_rgb = run.visualize_detections(run.detections, run.img, run.colors)
                    run.publish(viz_msg, viz_rgb)
                elif run.raf_state.visualize_detections == "selection" or run.raf_state.visualize_detections == "normal+selection":
                    item, mouse_pos, selection_msg = run.Selection()
                    # print("Selected Item: ", item)
                    selected_action = run.actionSelection(item, mouse_pos)
                    if selected_action == 'Cancel':
                        run.request_pub.publish('system-state-selection')
                        rospy.sleep(0.1)
                    elif selected_action is not None:
                        run.request_pub.publish('system-state-action')
                        rospy.sleep(0.1)
                        run.action_pub.publish(selected_action)
                        rospy.sleep(0.1)
                        run.selection_pub.publish(selection_msg)
                    else:
                        run.request_pub.publish('system-state-idle')
                        rospy.sleep(0.1)
                else:
                    im_rgb = cv2.cvtColor(run.img, cv2.COLOR_BGR2RGB)
                    im_msg = bridge.cv2_to_imgmsg(im_rgb, encoding="rgb8")
                    run.publish(im_msg, im_rgb)
            elif run.raf_state.enable_scene_detections:
                run.detections = run.scene_detections

                if run.detections is None:
                    continue

                # Visualize based on RAF state
                if run.raf_state.visualize_detections == "normal":
                    viz_msg, viz_rgb = run.visualize_detections(run.detections, run.img, run.colors)
                    run.publish(viz_msg, viz_rgb)
                elif run.raf_state.visualize_detections == "selection" or run.raf_state.visualize_detections == "normal+selection":
                    item, mouse_pos = run.Selection()
                    # print("Selected Item: ", item)
                    selected_action = run.actionSelection(item, mouse_pos)
                    if selected_action == 'Cancel':
                        run.request_pub.publish('system-state-selection')
                        rospy.sleep(0.1)
                    elif selected_action is not None:
                        run.request_pub.publish('system-state-action')
                        rospy.sleep(0.1)
                        run.action_pub.publish(selected_action)
                    else:
                        run.request_pub.publish('system-state-idle')
                        rospy.sleep(0.1)
                else:
                    im_rgb = cv2.cvtColor(run.img, cv2.COLOR_BGR2RGB)
                    im_msg = bridge.cv2_to_imgmsg(im_rgb, encoding="rgb8")
                    run.publish(im_msg, im_rgb)
            else:
                im_rgb = cv2.cvtColor(run.img, cv2.COLOR_BGR2RGB)
                im_msg = bridge.cv2_to_imgmsg(im_rgb, encoding="rgb8")
                run.publish(im_msg, im_rgb)
        elif (run.raf_state.system_state == 'action' or run.raf_state.system_state == 'idle') and run.raf_state.visualize_detections == "normal":
            if run.raf_state.enable_arm_detections:
                run.detections = run.arm_detections
            elif run.raf_state.enable_scene_detections:
                run.detections = run.scene_detections

            if run.detections is None:
                continue
            
            viz_msg, viz_rgb = run.visualize_detections(run.detections, run.img, run.colors)
            run.publish(viz_msg, viz_rgb)
                
        else:
            im_rgb = cv2.cvtColor(run.img, cv2.COLOR_BGR2RGB)
            im_msg = bridge.cv2_to_imgmsg(im_rgb, encoding="rgb8")
            run.publish(im_msg, im_rgb)

    return 0

if __name__ == '__main__':
    sys.exit(main())