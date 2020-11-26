#importing the required modules
import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
import numpy as np
import os, json, cv2, random, glob
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import os
import numpy as np
from detectron2.structures import BoxMode
import itertools
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import pkg_resources
import random
from detectron2.utils.visualizer import ColorMode
from flask import Flask, request, Response
from flask import Flask,jsonify,abort,make_response,request
import jsonpickle
from detectron2.utils.visualizer import (
    ColorMode,
    Visualizer,
    _create_text_labels,
    _PanopticPrediction,
)



#defining metadata with class labels

nom_dataset = "trash2_"

MetadataCatalog.get(nom_dataset).set(thing_classes=['Opaque Bottle', 'Cartoon Packaging',
                                                'Transparent Bottle', 'Couloured Plastic', 'Paper',
                                                'Glass', 'Metal', 'Transparent Plastic', 
                                                'Plastic Pot'])
    
metadata = MetadataCatalog.get(nom_dataset)








#setting configuration parameters
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.OUTPUT_DIR = "output"
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_best_8217.pth')#   
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # set the testing threshold for this model
predictor = DefaultPredictor(cfg)




# Initialize the Flask application
app = Flask(__name__)


# route http posts to this method
@app.route('/api/infer', methods=['POST'])
def infer():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    im = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    #saving image for debugging. disable it before deployment
    cv2.imwrite("client_image.jpg",im)

    #passing image to predictor

    outputs = predictor(im)
    
    
    #code for drawing mask of detected objects
    # v = Visualizer(im[:, :, ::-1],
    #                metadata=metadata , 
    #                scale=0.5, 
    #                instance_mode=ColorMode.IMAGE  # remove the colors of unsegmented pixels
    # )
    # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # scale_percent = 50 # percent of original size
    # width = int(im.shape[1] * scale_percent / 100)
    # height = int(im.shape[0] * scale_percent / 100)
    # dim = (width, height)
    # # resize image
    # resized = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
    # #print(resized.shape)
    # vis = np.concatenate((resized,v.get_image()[:, :, ::-1] ), axis=1)
    # output_path = "/home/ml/projects/ficha_utils/predictor/results/" + "005001.jpg"
    # cv2.imwrite(output_path, vis)


    #convert predictions to cpu format
    predictions = outputs["instances"].to("cpu")

    #extracting boxes, scores and classes

    boxes = predictions.pred_boxes.tensor.numpy() if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = predictions.pred_classes.numpy() if predictions.has("pred_classes") else None

    
    #print("Predictions", classes.to("cpu"))
    
    #stacking labels of detected objects with the classification confidence, api returns this data back
    labels = _create_text_labels(classes, scores, metadata.get("thing_classes", None))
    print("labels",labels)


    #converting predicted data to json format
    data = {}
    data['trash'] = labels
    return make_response(jsonify({'data': data}), 200)


# start flask app
app.run(host="0.0.0.0", port=5000)
    
