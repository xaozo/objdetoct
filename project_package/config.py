#from detectron2.config import CfgNode as CN
from detectron2.data.datasets import register_coco_instances
from datetime import datetime

def add_config(cfg):
    """
    Add config for model.
    """
    
    """
    register_coco_instances("train", {}, 
                            "./dataset/train/labels_train.json",
                            "./dataset/train/data")
    register_coco_instances("val", {},
                            "./dataset/val/labels_val.json",
                            "./dataset/val/data")
   
    """
    
    ### EPOCHS AND BATCH SIZE ###
    EPOCHS = 5000
    BATCH_SIZE = 16
    ##############
    
    # Define datasets
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ("val",)
    
    # Output folder
    cfg.OUTPUT_DIR = "./output/trial-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")    

    # Training hyperparameters
    cfg.SOLVER.IMS_PER_BATCH = BATCH_SIZE
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.MAX_ITER = BATCH_SIZE*EPOCHS
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS_BATCH_SIZE_PER_IMAGE = 64 # region proposals
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.RETINANET.NUM_CLASSES = 3
    
    cfg.TEST.EVAL_PERIOD = BATCH_SIZE # iteration interval to calculate val loss
    