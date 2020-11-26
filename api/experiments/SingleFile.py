import random
import cv2
import numpy as np
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

cfg = get_cfg()
#cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "/home/lmesdon/Ficha/model_0019999.pth"
cfg.OUTPUT_DIR = "results"
data = "test1.jpg"
predictor = DefaultPredictor(cfg)

im = cv2.imread(data)
outputs = predictor(im)
v = Visualizer(im[:, :, ::-1],
            metadata=MetadataCatalog.get("trash").set(thing_classes=['Opaque Bottle', 'Cartoon Packaging',
                                                  'Transparent Bottle', 'Couloured Plastic', 'Paper',
                                                  'Glass', 'Metal', 'Transparent Plastic', 
                                                  'Plastic Pot']) , 
            scale=0.5, 
            instance_mode=ColorMode.IMAGE  # remove the colors of unsegmented pixels
)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

scale_percent = 50 # percent of original size
width = int(im.shape[1] * scale_percent / 100)
height = int(im.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
#cv2.imwrite(d["file_name"].split("/")[1],v.get_image()[:, :, ::-1])
#img1 = cv2.imread('img1.png')
#img2 = cv2.imread('img2.png')
vis = np.concatenate((resized,v.get_image()[:, :, ::-1] ), axis=1)
output_path = "/home/lmesdon/Ficha/results/" + data
cv2.imwrite(output_path, vis)