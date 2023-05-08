from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
# from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
import os

# Initialize predictor
CONFIDENCE_THRESHOLD = .5

def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "/home/labuser/ros_ws/src/odhe_ros/arm_camera_dataset2/output_50k/model_final.pth")
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = CONFIDENCE_THRESHOLD
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 16
    cfg.freeze()
    return cfg

if __name__ == "__main__":
    cfg = setup_cfg()
    register_coco_instances("train_set", {}, "/home/labuser/ros_ws/src/odhe_ros/arm_camera_dataset2/train/annotations.json", "/home/labuser/ros_ws/src/odhe_ros/arm_camera_dataset2/train")
    train_metadata = MetadataCatalog.get("train_set")

    train_metadata.thing_classes = ["Gripper", "Beans", "Cottage Cheese", "Yogurt", "Watermelon", "Banana", \
                                                      "Strawberry", "Celery", "Carrot", "Pretzel", "Knife", "Spoon", "Fork", \
                                                      "Cup", "Bowl", "Plate"]

    train_metadata.thing_colors = [(178, 80, 80), (140, 120, 240), (89, 134, 179), (250, 250, 55), (131, 224, 112), \
                                                     (255, 204, 51), (50, 183, 250), (102, 255, 102), (184, 61, 245), (36, 179, 83), \
                                                     (221, 255, 51), (255, 96, 55), (255, 0, 124), (52, 209, 183), (250, 50, 83), (51, 221, 255)]

    input = "/home/labuser/ros_ws/src/odhe_ros/arm_camera_dataset2/demo_images/03.jpg"
    img = read_image(input, format="BGR")
    predictor = DefaultPredictor(cfg)
    predictions = predictor(img)
    instances = predictions['instances']
    pred_boxes = instances.pred_boxes
    scores = instances.scores
    pred_classes = instances.pred_classes
    for box, score, label in zip(pred_boxes, scores, pred_classes):
        name = train_metadata.thing_classes[label]
        print(box.tolist(), float(score), name)