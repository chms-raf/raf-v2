import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import time

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode

import rospy
import sys
from cv_bridge import CvBridge
from raf.msg import Result
from sensor_msgs.msg import Image, RegionOfInterest

class maskRCNN(object):
    def __init__(self):
        # Params
        self.image = None
        self.br = CvBridge()

        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(15)

        # Publishers
        self.pub = rospy.Publisher('arm_camera_objects', Image, queue_size=10)
        self.result_pub = rospy.Publisher('arm_camera_results', Result, queue_size=10)

        # Subscribers
        rospy.Subscriber("/camera/color/image_raw", Image, self.callback)

    def callback(self, msg):
        self.image = self.convert_to_cv_image(msg)
        self._header = msg.header

    def get_img(self):
        result = self.image
        return result

    def getResult(self, predictions, classes):

        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            #print(type(masks))
        else:
            return

        result_msg = Result()
        result_msg.header = self._header
        result_msg.class_ids = predictions.pred_classes if predictions.has("pred_classes") else None
        result_msg.class_names = np.array(classes)[result_msg.class_ids.numpy()]
        result_msg.scores = predictions.scores if predictions.has("scores") else None

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            mask = np.zeros(masks[i].shape, dtype="uint8")
            mask[masks[i, :, :]]=255
            mask = self.br.cv2_to_imgmsg(mask)
            result_msg.masks.append(mask)

            box = RegionOfInterest()
            box.x_offset = np.uint32(x1)
            box.y_offset = np.uint32(y1)
            box.height = np.uint32(y2 - y1)
            box.width = np.uint32(x2 - x1)
            result_msg.boxes.append(box)

        return result_msg

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

    def publish(self, img, result_msg):
        self.pub.publish(img)
        self.result_pub.publish(result_msg)
        self.loop_rate.sleep()


def main():
    """ Mask RCNN Object Detection with Detectron2 """
    rospy.init_node("mask_rcnn", anonymous=True)
    bridge = CvBridge()
    start_time = time.time()
    image_counter = 0
    
    register_coco_instances("train_set", {}, "/home/labuser/ros_ws/src/raf/arm_camera_dataset2/train/annotations.json", "/home/labuser/ros_ws/src/raf/arm_camera_dataset2/train")
    train_metadata = MetadataCatalog.get("train_set")
    print(train_metadata)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 16  # 16 classes

    # Temporary Solution. If I train again I think I can use the dynamically set path again
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "/home/labuser/ros_ws/src/raf/arm_camera_dataset2/models/final_model/model_final.pth")
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9   # set the testing threshold for this model
    predictor = DefaultPredictor(cfg)

    # class_names = MetadataCatalog.get("train_set").thing_classes

    class_names = ['Plate', 'Bowl', 'Cup', 'Fork', 'Spoon', 'Knife', 'Pretzel', 'Carrot', 'Celery', 
                   'Strawberry', 'Banana', 'Watermelon', 'Yogurt', 'Cottage Cheese', 'Beans', 'Gripper']

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
    colors = list([color_plate, color_bowl, color_cup, color_fork, color_spoon, color_knife, 
                   color_pretzel, color_carrot, color_celery, color_strawberry, color_banana, 
                   color_watermelon, color_yogurt, color_cottage_cheese, color_beans, color_gripper])

    alpha = .4

    run = maskRCNN()
    while not rospy.is_shutdown():
        # Get images
        img = run.get_img()

        if img is not None:
            outputs = predictor(img)
            predictions = outputs["instances"].to("cpu")

            # Get results
            result = run.getResult(predictions, class_names)

            # Visualize using custom cv2 code
            if result is not None:
                result_cls = result.class_names
                result_clsId = result.class_ids
                result_scores = result.scores
                result_masks = result.masks

                # Create copies of the original image
                im = img.copy()
                output = img.copy()

                # Initialize lists
                masks = []
                masks_indices = []
                for i in range(len(result_clsId)):
                    # Obtain current object mask as a numpy array (black and white mask of single object)
                    current_mask = bridge.imgmsg_to_cv2(result_masks[i])

                    # Find current mask indices
                    mask_indices = np.where(current_mask==255)

                    # Add to mask indices list
                    if len(masks_indices) > len(result_clsId):
                        masks_indices = []
                    else:
                        masks_indices.append(mask_indices)

                    # Add to mask list
                    if len(masks) > len(result_clsId):
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

                for i in range(len(result_clsId)):
                    # Select correct object color
                    color = colors[result_clsId[i]]

                    # Change the color of the current mask object
                    im[masks_indices[i][0], masks_indices[i][1], :] = color

                # Apply alpha scaling to image to adjust opacity
                cv2.addWeighted(im, alpha, output, 1 - alpha, 0, output)

                for i in range(len(result_clsId)):
                    # Draw Bounding boxes
                    start_point = (result.boxes[i].x_offset, result.boxes[i].y_offset)
                    end_point = (result.boxes[i].x_offset + result.boxes[i].width, result.boxes[i].y_offset + result.boxes[i].height)
                    start_point2 = (result.boxes[i].x_offset + 2, result.boxes[i].y_offset + 2)
                    end_point2 = (result.boxes[i].x_offset + result.boxes[i].width - 2, result.boxes[i].y_offset + 12)
                    color = colors[result_clsId[i]]
                    box_thickness =  1

                    name = result_cls[i]
                    score = result_scores[i]
                    conf = round(score.item() * 100, 1)
                    string = str(name) + ":" + str(conf) + "%"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    org = (result.boxes[i].x_offset + 2, result.boxes[i].y_offset + 10)
                    fontScale = .3
                    text_thickness = 1
                    output = cv2.rectangle(output, start_point, end_point, color, box_thickness)
                    output = cv2.rectangle(output, start_point2, end_point2, color, -1)     # Text box
                    output = cv2.putText(output, string, org, font, fontScale, [0, 0, 0], text_thickness, cv2.LINE_AA, False)

                im_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                im_msg = bridge.cv2_to_imgmsg(im_rgb, encoding="rgb8")


            # Display Image Counter
            image_counter = image_counter + 1
            if (image_counter % 11) == 10:
                rospy.loginfo("Images detected per second=%.2f", float(image_counter) / (time.time() - start_time))

            run.publish(im_msg, result)    
        

    return 0

if __name__ == '__main__':
    sys.exit(main())