import os
import ssl
import certifi
import numpy as np
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances

import wandb

# Set up logger for Detectron2
setup_logger()

# SSL context for secure connections
def create_context():
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_verify_locations(certifi.where())
    return context

# Display metadata for verification
print('\n METADATA CATALOG')
print(list(detectron2.data.MetadataCatalog))

# Registering the COCO dataset for training and validation
register_coco_instances(
    "src/data-coco/annotations/json_annotation_train.json", 
    {}, 
    "src/data-coco/annotations/json_annotation_train.json", 
    "src/data-coco/images/train"
)
register_coco_instances(
    "src/data-coco/annotations/json_annotation_val.json", 
    {}, 
    "src/data-coco/annotations/json_annotation_val.json", 
    "src/data-coco/images/val"
)

# Display metadata for verification
print('\n METADATA CATALOG')
print(list(detectron2.data.MetadataCatalog))

# Configure Detectron2 model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("src/data-coco/annotations/json_annotation_train.json")
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 20    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.MODEL.DEVICE = "cpu"  # Force CPU usage

# TODO: change root output to src/retinaNet/output
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


# SSL configuration for urllib 
ssl._create_default_https_context = create_context()

api = wandb.Api()

run = api.run("neuropoly-axon-detection/retinanet-project/<run_id>")
if run.state == "finished":
    for i, row in run.history().iterrows():
      print(row["_timestamp"], row["accuracy"])


if __name__ == '__main__':
    # Train the model
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    try:
        trainer.train()

    except Exception as e:
        print('Training stopped due to:' + str(e))
