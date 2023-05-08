#!/usr/bin/env python

# import torch, torchvision
# import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, cv2, random
import time
import pyrealsense2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode

import rospy, sys
from cv_bridge import CvBridge
from raf.msg import Result
from gpd_ros.msg import CloudSamples, CloudSources, CloudIndexed
from sensor_msgs.msg import Image, RegionOfInterest, PointCloud2, CameraInfo
from geometry_msgs.msg import Point
from std_msgs.msg import Int64

class maskRCNN(object):
    def __init__(self):
        # Params
        self.image = None
        self.br = CvBridge()

        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(15)

        # Subscribers
        # Scene Camera
        # rospy.Subscriber("/scene_camera/color/image_raw", Image, self.callback)
        # rospy.Subscriber("/scene_camera/depth/color/points", PointCloud2, self.pc_callback)
        # rospy.Subscriber("/scene_camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        # rospy.Subscriber("/scene_camera/color/camera_info", CameraInfo, self.camInfo_callback)

        # Arm Camera
        rospy.Subscriber("/camera/color/image_raw", Image, self.callback)
        rospy.Subscriber("/camera/depth_registered/points", PointCloud2, self.pc_callback)
        # rospy.Subscriber("/camera/depth_registered/sw_registered/image_rect", Image, self.depth_callback)
        rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)
        rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camInfo_callback)

        # Publishers
        self.pub = rospy.Publisher('arm_camera_objects', Image, queue_size=10)
        self.result_pub = rospy.Publisher('arm_camera_results', Result, queue_size=10)
        self.pc_pub = rospy.Publisher('cloud_samples', CloudSamples, queue_size=10)
        self.pc_pub2 = rospy.Publisher('cloud_indices', CloudIndexed, queue_size=10)

        self.depth_pub = rospy.Publisher('arm_camera_objects_depth', Image, queue_size=10)

    def callback(self, msg):
        self.image = self.convert_to_cv_image(msg)
        self._header = msg.header

    def pc_callback(self, msg):
        self.pointCloud = msg

    def depth_callback(self, msg):
        self.depth_array = self.convert_depth_image(msg)
        self._header = msg.header

    def convert_depth_image(self, image_msg):
        if image_msg is not None:
            depth_image = self.br.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
            depth_array = np.asarray(depth_image, dtype=np.float32)
            # depth_array = np.array(depth_image, dtype=np.float32)
            return depth_array
        else:
            return None

    def get_img(self):
        result = self.image
        return result

    def get_depth_array(self):
        result = self.depth_array
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

    def sort_detections(self, msg):
        # Sort detections by y-position of upper left bbox corner
        # TODO: Sort by y-position first and then sort again by x-position
        # This will prevent object ids from flipping back and forth if they are at the same y-position

        target = self.Out_transfer(msg.class_ids, msg.class_names, msg.scores, msg.boxes, msg.masks)

        # Sort by y-offset
        # self.Sort_quick(target, 0, len(target)-1)

        # Sort by y-offset and x-offset
        self.Sort_quick(target, 0, len(target)-1, y=True)

        #after sort the y, then start sorting the x:
        arr_y = [(target[w][3].y_offset + target[w][3].y_offset + target[w][3].height)/2 for w in range(len(target))] #(y1+y2)/2

        store = []
        for i in range(len(arr_y)):
            if arr_y.count(arr_y[i]) > 1:
                store.append([i, arr_y.count(arr_y[i])+1])

        if len(store) !=0:
            for each_group in store:
                self.Sort_quick(target, each_group[0], each_group[1], y=False)

        return target

    def Out_transfer(self, class_id, class_name, score, box, mask):

        num = int(len(class_id))
        target = []

        for i in range(num):

            target.append([class_id[i], class_name[i], score[i], box[i], mask[i]])

        return target

    def partition(self, target, low, high, y=True):

        i = ( low-1 )
        arr = []
        # pdb.set_trace()
        if y:
            # x1 = target[w][3].x_offset
            # y1 = target[w][3].y_offset
            # x2 = target[w][3].x_offset + target[w][3].width
            # y2 = target[w][3].y_offset + target[w][3].height
            arr = [(target[w][3].y_offset + target[w][3].y_offset + target[w][3].height)/2 for w in range(len(target))] #box:[x1, y1, x2, y2]  value :(y1+y2)/2
        else:
            arr = [(target[w][3].x_offset + target[w][3].x_offset + target[w][3].width)/2 for w in range(len(target))] #box:[x1, y1, x2, y2]  value :(x1+x2)/2

        pivot = arr[high]

        for j in range(low , high): 
            if   arr[j] <= pivot: 
                i = i+1 
                target[i],target[j] = target[j],target[i] 
    
        target[i+1],target[high] = target[high],target[i+1] 

        return ( i+1 )

    def Sort_quick(self, target, low, high, y):

        if low < high: 
            pi = self.partition(target,low,high, y) 
    
            self.Sort_quick(target, low, pi-1, y) 
            self.Sort_quick(target, pi+1, high, y)

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

    def camInfo_callback(self, msg):
        self.cam_info = msg

    def convert_depth_to_phys_coord_using_realsense(self, x, y, depth, cameraInfo):
        _intrinsics = pyrealsense2.intrinsics()
        _intrinsics.width = cameraInfo.width
        _intrinsics.height = cameraInfo.height
        _intrinsics.ppx = cameraInfo.K[2]
        _intrinsics.ppy = cameraInfo.K[5]
        _intrinsics.fx = cameraInfo.K[0]
        _intrinsics.fy = cameraInfo.K[4]
        #_intrinsics.model = cameraInfo.distortion_model
        _intrinsics.model  = pyrealsense2.distortion.none
        _intrinsics.coeffs = [i for i in cameraInfo.D]
        result = pyrealsense2.rs2_deproject_pixel_to_point(_intrinsics, [x, y], depth)
        # result[0]: right, result[1]: down, result[2]: forward
        return result[0], result[1], result[2]
        # return result[2], -result[0], -result[1]

    def publish(self, img, img_depth, result_msg, cloud_samples_msg, cloud_indices_msg):
        self.pub.publish(img)
        self.depth_pub.publish(img_depth)
        self.result_pub.publish(result_msg)
        self.pc_pub.publish(cloud_samples_msg)
        self.pc_pub2.publish(cloud_indices_msg)
        self.loop_rate.sleep()


def main():
    """ Mask RCNN Object Detection with Detectron2 """
    rospy.init_node("mask_rcnn", anonymous=True)
    bridge = CvBridge()
    start_time = time.time()
    image_counter = 0
    
    register_coco_instances("train_set", {}, "/home/labuser/ros_ws/src/raf/arm_camera_dataset2/train/annotations.json", "/home/labuser/ros_ws/src/raf/arm_camera_dataset2/train")
    # register_coco_instances("test_set", {}, "/home/labuser/ros_ws/src/raf/arm_camera_dataset2/test/annotations.json", "/home/labuser/ros_ws/src/raf/arm_camera_dataset2/test")
    
    # train_metadata = MetadataCatalog.get("train_set")
    # print(train_metadata)
    # dataset_dicts_train = DatasetCatalog.get("train_set")

    # test_metadata = MetadataCatalog.get("test_set")
    # print(test_metadata)
    # dataset_dicts_test = DatasetCatalog.get("test_set")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # cfg.DATASETS.TRAIN = ("train_set")
    # cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
    # cfg.DATALOADER.NUM_WORKERS = 4
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # initialize from model zoo
    # cfg.SOLVER.IMS_PER_BATCH = 4
    # cfg.SOLVER.BASE_LR = 0.01
    # cfg.SOLVER.MAX_ITER = 1000 # 300 iterations seems good enough, but you can certainly train longer
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (128)  # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 16  # 16 classes

    # Temporary Solution. If I train again I think I can use the dynamically set path again
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "/home/labuser/ros_ws/src/raf/arm_camera_dataset2/models/final_model/model_final.pth")
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9   # set the testing threshold for this model
    # cfg.DATASETS.TEST = ("test_set")
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

    rospy.sleep(5.0)

    run = maskRCNN()
    while not rospy.is_shutdown():
        # Publish
        try:
            run.publish(im_msg, im_depth_msg, result, cloud_samples, cloud_indices)
        except:
            pass

        # Get images
        img = run.get_img()

        if img is None:
            continue

        outputs = predictor(img)
        predictions = outputs["instances"].to("cpu")

        # Get results
        unsorted = run.getResult(predictions, class_names)

        # Sort detections by x and y
        sorted = run.sort_detections(unsorted)

        result = Result()
        for i in range(len(sorted)):
            result.class_ids.append(sorted[i][0])
            result.class_names.append(sorted[i][1])
            result.scores.append(sorted[i][2])
            result.boxes.append(sorted[i][3])
            result.masks.append(sorted[i][4])

        # Visualize using detectron2 built in visualizer
        # v = Visualizer(im[:, :, ::-1],
        #             metadata=train_metadata, 
        #             scale=1.0 
        #             # instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
        # )
        # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # im = v.get_image()[:, :, ::-1]
        # im_msg = bridge.cv2_to_imgmsg(im, encoding="bgr8")

        # Visualize using custom cv2 code
        if result is None:
            continue

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
        for i in range(len(result_cls)):
            # Obtain current object mask as a numpy array (black and white mask of single object)
            current_mask = bridge.imgmsg_to_cv2(result_masks[i])

            # Find current mask indices
            mask_indices = np.where(current_mask==255)

            # Add to mask indices list
            if len(masks_indices) > len(result_cls):
                masks_indices = []
            else:
                masks_indices.append(mask_indices)

            # Add to mask list
            if len(masks) > len(result_cls):
                masks = []
            else:
                masks.append(current_mask)

        if len(masks) > 0:
            # Create composite mask
            composite_mask = sum(masks)

            # Clip composite mask between 0 and 255   
            composite_mask = composite_mask.clip(0, 255)

        idx = [ii for ii, e in enumerate(class_names) if e == result_cls[i] ] # If it is a cup

        for i in range(len(result_cls)):
            # Select correct object color
            color = colors[class_names.index(result_cls[i])]

            # Change the color of the current mask object
            im[masks_indices[i][0], masks_indices[i][1], :] = color

        # Apply alpha scaling to image to adjust opacity
        cv2.addWeighted(im, alpha, output, 1 - alpha, 0, output)

        for i in range(len(result_cls)):
            # Draw Bounding boxes
            start_point = (result.boxes[i].x_offset, result.boxes[i].y_offset)
            end_point = (result.boxes[i].x_offset + result.boxes[i].width, result.boxes[i].y_offset + result.boxes[i].height)
            start_point2 = (result.boxes[i].x_offset + 2, result.boxes[i].y_offset + 2)
            end_point2 = (result.boxes[i].x_offset + result.boxes[i].width - 2, result.boxes[i].y_offset + 12)
            color = colors[class_names.index(result_cls[i])]
            box_thickness =  1
            centroid = (int((start_point[0] + end_point[0]) / 2), int((start_point[1] + end_point[1]) / 2) + 60)
            if i == 0:
                start_point_cup = start_point
                end_point_cup = end_point
                centroid_cup = centroid

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

        ##### The entire goal of the below code is to get N random points on the mask in 3D
        ##### and publish on cloud samples topic for GPD
        item_names = result_cls
        idx = [i for i, e in enumerate(item_names) if e == "Cup" ] # If it is a cup

        # print("Detected Items: ", item_names)
        # print("-------------------\n")

        if len(idx) < 1:
            continue

        mask = bridge.imgmsg_to_cv2(result_masks[idx[0]]) #TODO: Handle multiple cups
        coord = cv2.findNonZero(mask)   # Coordinates of the mask that are on the food item
        # coord = np.nonzero(mask)

        # Pick random points on the object mask
        sample_list = list()
        samples = np.array([[centroid_cup[0], centroid_cup[1]], 
                            [centroid_cup[0] + 30, centroid_cup[1]], 
                            [centroid_cup[0] - 30, centroid_cup[1]], 
                            [centroid_cup[0], centroid_cup[1] + 30], 
                            [centroid_cup[0], centroid_cup[1] - 30],
                            [centroid_cup[0], centroid_cup[1] - 120], 
                            [centroid_cup[0] + 30, centroid_cup[1] - 120], 
                            [centroid_cup[0] - 30, centroid_cup[1] - 120], 
                            [centroid_cup[0], centroid_cup[1] + 30 - 120], 
                            [centroid_cup[0], centroid_cup[1] - 30 - 120]])

        img_depth = run.get_depth_array()
        output_depth = img_depth.copy()

        scaleX = 0.375
        scaleY = 0.375
        biasX = 7 #TODO: needs to be aligned properly
        biasY = 45 #TODO: needs to be aligned properly

        for ii in range(len(samples)):
            point = Point()
            # x = random.choice(coord[0,0])
            # y = random.choice(coord[0,1])
            # x = random.choice(coord[:,0,0])
            # y = random.choice(coord[:,0,1])
            x = samples[ii,0]
            y = samples[ii,1]
            x_depth = int((x * scaleX) + biasX)
            y_depth = int((y * scaleY) + biasY)
            start_point_cup_depth_x = int((start_point_cup[0] * scaleX) + biasX)
            start_point_cup_depth_y = int((start_point_cup[1] * scaleY) + biasY)
            end_point_cup_depth_x = int((end_point_cup[0] * scaleX) + biasX)
            end_point_cup_depth_y = int((end_point_cup[1] * scaleY) + biasY)
            start_point_cup_depth = (start_point_cup_depth_x, start_point_cup_depth_y)
            end_point_cup_depth = (end_point_cup_depth_x, end_point_cup_depth_y)

            # TODO: Current problem is can't subscribe to depth registered topic because lot's of NaN's
            # TODO: So have to subscribe to offset topic instead
            # TODO: Possible solution is to use np.nan_to_num to fix depth cloud

            # print(f"[X: {x}, Y: {y}]")
            output = cv2.circle(output, (x, y), 3, [51, 204, 255], -1)
            output_depth = cv2.circle(output_depth, (x_depth, y_depth), 2, [0, 0, 0], -1)
            output_depth = cv2.rectangle(output_depth, start_point_cup_depth, end_point_cup_depth, color, box_thickness)
            # depth = (run.depth_array[y, x]) / 1000
            # depth = (run.depth_array[135, 240]) / 1000
            # print(run.depth_array.shape)
            # print(type(run.depth_array))
            # depth = (run.depth_array[360, 640])
            # depth = (run.depth_array[int(x), int(y)]) / 1000 # x and y switched
            depth = (run.depth_array[y_depth, x_depth]) / 1000 # x and y switched
            # depth = .65
            # Deproject pixels and depth to 3D coordinates (camera frame)
            X, Y, Z = run.convert_depth_to_phys_coord_using_realsense(x, y, depth, run.cam_info)
            # print("(x,y,z) to convert: ("+str(x)+", "+str(y)+", "+str(depth)+")")
            # print("(X,Y,Z) converted: ("+str(X)+", "+str(Y)+", "+str(Z)+")")
            # print("Depth: ", Z)
            point.x = X; point.y = Y; point.z = Z
            if Z is not None and Z > 0.05:
                sample_list.append(point)

        samples = []

        print("Samples in List: ", len(sample_list))
        for sample in sample_list:
            print(f"[x: {sample.x}, y: {sample.y}, z: {sample.z}]")
        print("------------------------------------\n")

        cam_source = Int64()
        cam_source.data = 0

        cloud_source = CloudSources()
        # try:
        #     cloud_source.cloud = run.pointCloud
        # except:
        #     pass
        
        cloud_source.cloud = run.pointCloud
        cloud_source.camera_source = [cam_source]
        view_point = Point()
        # view_point.x = 0.640; view_point.y = 0.828; view_point.z = 0.505
        view_point.x = 0; view_point.y = 0; view_point.z = 0
        cloud_source.view_points = [view_point]

        cloud_samples = CloudSamples()
        cloud_samples.cloud_sources = cloud_source
        cloud_samples.samples = sample_list

        cloud_indices = CloudIndexed()
        cloud_indices.cloud_sources = cloud_source
        cloud_indices.indices = [55, 1569, 8456, 12579, 16789]

        # Print publish info
        # print(type(cloud_source.cloud))
        # print(cloud_source.camera_source)
        # print(cloud_source.view_points)
        # print("")
        # print(type(cloud_samples.cloud_sources))
        # print(cloud_samples.samples)
        # print("-------------------------\n")

        # Display Image Counter
        # image_counter = image_counter + 1
        # if (image_counter % 11) == 10:
        #     rospy.loginfo("Images detected per second=%.2f", float(image_counter) / (time.time() - start_time))

        im_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        im_msg = bridge.cv2_to_imgmsg(im_rgb, encoding="rgb8")
        im_depth_msg = bridge.cv2_to_imgmsg(output_depth)
        # run.publish(im_msg, result, cloud_samples, cloud_indices)    
        

    return 0

if __name__ == '__main__':
    sys.exit(main())