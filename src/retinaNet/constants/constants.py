# Paths for COCO dataset
COCO_TRAIN_ANNOTATION = "../data-coco/annotations/json_annotation_train.json"
COCO_TRAIN_IMAGES = "../data-coco/images/train"
COCO_VAL_ANNOTATION = "../data-coco/annotations/json_annotation_val.json"
COCO_VAL_IMAGES = "../data-coco/images/val"

# Detectron2 configuration
CONFIG_FILE = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
OUTPUT_DIR = "src/retinaNet/output"

# WandB configuration
WANDB_ENTITY = "neuropoly-axon-detection"
WANDB_PROJECT = "retinanet-project"
WANDB_RUN_NAME = "retinaRun"
WANDB_RUN_ID = "0001"