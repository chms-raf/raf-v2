import torch
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetMapper

import pickle

# LossEvalHook Imports
from detectron2.engine.hooks import HookBase
import time
import datetime
import logging
from detectron2.utils.logger import log_every_n_seconds
import numpy as np
import detectron2.utils.comm as comm

print(cv2.getBuildInformation())

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
                     
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))
        # Uncomment if using multiple GPUs
        # hooks = hooks[:-2] + hooks[-2:][::-1]
        return hooks

class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
    
    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
            
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):            
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses
            
    def _get_loss(self, data):
        # How loss is calculated on train_loop 
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
        
        
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)

# Register the dataset
register_coco_instances("train_set", {}, "/home/labuser/ros_ws/src/raf/arm_camera_dataset2/train/annotations_food.json", "/home/labuser/ros_ws/src/raf/arm_camera_dataset2/train")
register_coco_instances("test_set", {}, "/home/labuser/ros_ws/src/raf/arm_camera_dataset2/validation/annotations_food.json", "/home/labuser/ros_ws/src/raf/arm_camera_dataset2/validation")

train_metadata = MetadataCatalog.get("train_set")
print(train_metadata)
dataset_dicts = DatasetCatalog.get("train_set")

test_metadata = MetadataCatalog.get("test_set")
print(test_metadata)
dataset_dicts_test = DatasetCatalog.get("test_set")

# Visualize Training data
for d in random.sample(dataset_dicts, 5):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=train_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imshow('Training Image', vis.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# # Visualize Test data
for d in random.sample(dataset_dicts_test, 5):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=test_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imshow('Test Image', vis.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Train the model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("train_set",)
cfg.DATASETS.TEST = ("test_set",)  # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # initialize from model zoo
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 10000 # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
cfg.TEST.EVAL_PERIOD = 100

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = DefaultTrainer(cfg)
trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

print("Model trained.")

checkpointer = DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR)
# torch.save(trainer.model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "test.pth"))
# torch.save(trainer.model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "test.pkl"))

f = open("model_final_dict.pkl","wb")
# This line below does not result in the right file type
# It should be a dict, but instead is a collections.ordered_dict
# Maybe the line below should instead be:
#   pickle.dump(trainer.model, f)
pickle.dump(trainer.model.state_dict(), f)
f.close()

f = open("model_final.pkl","wb")
pickle.dump(trainer.model, f)
f.close()

print("Model saved.")

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9   # set the testing threshold for this model
cfg.DATASETS.TEST = ("test_set")
cfg.TEST.EVAL_PERIOD = 100
predictor = DefaultPredictor(cfg)

# for d in random.sample(dataset_dicts_test, 5):    
#     im = cv2.imread(d["file_name"])
#     outputs = predictor(im)
#     v = Visualizer(im[:, :, ::-1],
#                    metadata=test_metadata, 
#                    scale=0.8, 
#                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
#     )
#     v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     cv2.imshow('Output Image', v.get_image()[:, :, ::-1])
#     cv2.waitKey(0)

# Evaluate Performance
evaluator = COCOEvaluator("test_set", ("bbox", "segm"), False, output_dir="./output/")
test_loader = build_detection_test_loader(cfg, "test_set")
print(inference_on_dataset(trainer.model, test_loader, evaluator))
# # another equivalent way to evaluate the model is to use `trainer.test`