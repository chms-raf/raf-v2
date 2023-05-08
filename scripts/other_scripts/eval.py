from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# Register the dataset
register_coco_instances("train_set", {}, "/home/labuser/ros_ws/src/raf/arm_camera_dataset2/train/annotations_food.json", "/home/labuser/ros_ws/src/raf/arm_camera_dataset2/train")
register_coco_instances("test_set", {}, "/home/labuser/ros_ws/src/raf/arm_camera_dataset2/test/annotations_food.json", "/home/labuser/ros_ws/src/raf/arm_camera_dataset2/test")

train_metadata = MetadataCatalog.get("train_set")
# train_metadata.thing_classes = ["Gripper", "Beans", "Cottage Cheese", "Yogurt", "Watermelon", "Banana", \
#                                                       "Strawberry", "Celery", "Carrot", "Pretzel", "Knife", "Spoon", "Fork", \
#                                                       "Cup", "Bowl", "Plate"]

# train_metadata.thing_colors = [(178, 80, 80), (140, 120, 240), (89, 134, 179), (250, 250, 55), (131, 224, 112), \
#                                                     (255, 204, 51), (50, 183, 250), (102, 255, 102), (184, 61, 245), (36, 179, 83), \
#                                                     (221, 255, 51), (255, 96, 55), (255, 0, 124), (52, 209, 183), (250, 50, 83), (51, 221, 255)]
print(train_metadata)
dataset_dicts = DatasetCatalog.get("train_set")

test_metadata = MetadataCatalog.get("test_set")
# test_metadata.thing_classes = ["Gripper", "Beans", "Cottage Cheese", "Yogurt", "Watermelon", "Banana", \
#                                                       "Strawberry", "Celery", "Carrot", "Pretzel", "Knife", "Spoon", "Fork", \
#                                                       "Cup", "Bowl", "Plate"]

# test_metadata.thing_colors = [(178, 80, 80), (140, 120, 240), (89, 134, 179), (250, 250, 55), (131, 224, 112), \
#                                                     (255, 204, 51), (50, 183, 250), (102, 255, 102), (184, 61, 245), (36, 179, 83), \
#                                                     (221, 255, 51), (255, 96, 55), (255, 0, 124), (52, 209, 183), (250, 50, 83), (51, 221, 255)]
print(test_metadata)
dataset_dicts_test = DatasetCatalog.get("test_set")

# Visualize Training data
# for d in random.sample(dataset_dicts, 3):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=train_metadata, scale=0.5)
#     vis = visualizer.draw_dataset_dict(d)
#     cv2.imshow('Training Image', vis.get_image()[:, :, ::-1])
#     cv2.waitKey(0)

# # Visualize Test data
# for d in random.sample(dataset_dicts_test, 3):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=test_metadata, scale=0.5)
#     vis = visualizer.draw_dataset_dict(d)
#     cv2.imshow('Test Image', vis.get_image()[:, :, ::-1])
#     cv2.waitKey(0)

# Train the model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("train_set")
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # initialize from model zoo
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 300  # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8

# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0034999.pth")
cfg.MODEL.WEIGHTS = '/home/labuser/ros_ws/src/raf/arm_camera_dataset2/models/model_food/Huimings_Model/model_final.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9   # set the testing threshold for this model
cfg.DATASETS.TEST = ("test_set")
predictor = DefaultPredictor(cfg)

for d in random.sample(dataset_dicts_test, 3):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=test_metadata, 
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow('Output Image', v.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Evaluate Performance
evaluator = COCOEvaluator("test_set", ("bbox", "segm"), False, output_dir="./output/")
# evaluator = COCOEvaluator("test_set", tuple('bbox'), False, output_dir="./output/")
test_loader = build_detection_test_loader(cfg, "test_set")
print(inference_on_dataset(predictor.model, test_loader, evaluator))
# # another equivalent way to evaluate the model is to use `trainer.test`