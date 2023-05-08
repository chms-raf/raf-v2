#!/usr/bin/env python3

import numpy as np
import rospy, os, sys, cv2

from raf.msg import DetectionList, RafState
from sensor_msgs.msg import Image, RegionOfInterest
from cv_bridge import CvBridge

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
setup_logger()

class maskRCNN(object):
    def __init__(self):
        # Params
        self.br = CvBridge()
        self.image = None
        self.raf_state = None

        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(15)

        # Subscribers
        rospy.Subscriber("/camera/color/image_raw", Image, self.callback)
        rospy.Subscriber("/raf_state", RafState, self.state_callback)

        # Publishers
        self.pub = rospy.Publisher('arm_camera_detections', DetectionList, queue_size=10)

    def callback(self, msg):
        self.image = self.convert_to_cv_image(msg)
        self._header = msg.header

    def state_callback(self, msg):
        self.raf_state = msg

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

    def get_img(self):
        result = self.image
        return result

    def build_detection_msg(self, predictions, classes):

        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            #print(type(masks))
        else:
            return

        result_msg = DetectionList()
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

    def publish(self, detection_msg):
        self.pub.publish(detection_msg)
        self.loop_rate.sleep()

def main():
    """ Mask RCNN Object Detection with Detectron2 """
    rospy.init_node("arm_cam_detections", anonymous=True)

    register_coco_instances("train_set", {}, "/home/labuser/ros_ws/src/raf/arm_camera_dataset2/train/annotations.json", "/home/labuser/ros_ws/src/raf/arm_camera_dataset2/train")
    # train_metadata = MetadataCatalog.get("train_set")
    # print(train_metadata)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 16  # 16 classes

    # Temporary Solution. If I train again I think I can use the dynamically set path again
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "/home/labuser/ros_ws/src/raf/arm_camera_dataset2/models/final_model/model_final.pth")
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8   # set the testing threshold for this model
    predictor = DefaultPredictor(cfg)

    # class_names = MetadataCatalog.get("train_set").thing_classes

    class_names = ['Plate', 'Bowl', 'Cup', 'Fork', 'Spoon', 'Knife', 'Pretzel', 'Carrot', 'Celery', 
                   'Strawberry', 'Banana', 'Watermelon', 'Yogurt', 'Cottage Cheese', 'Beans', 'Gripper']

    run = maskRCNN()
    rospy.loginfo("Running Inference...")
    while not rospy.is_shutdown():
        # Get image
        img = run.get_img()

        if img is None:
            continue

        if run.raf_state is None:
            continue

        if run.raf_state.enable_arm_detections:
            rospy.loginfo_once("Running arm detection inference...")
            outputs = predictor(img)
            predictions = outputs["instances"].to("cpu")
            detection_msg = run.build_detection_msg(predictions, class_names)
            run.publish(detection_msg)
        else:
            detection_msg = DetectionList()
            run.publish(detection_msg)

if __name__ == '__main__':
    sys.exit(main())