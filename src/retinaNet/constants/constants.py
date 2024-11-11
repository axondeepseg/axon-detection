# Paths for COCO dataset
COCO_TRAIN_ANNOTATION = "data-coco/annotations/json_annotation_train.json"
COCO_TRAIN_IMAGES = "data-coco/images/train"
COCO_VAL_ANNOTATION = "data-coco/annotations/json_annotation_val.json"
COCO_VAL_IMAGES = "data-coco/images/val"
COCO_TEST_ANNOTATION = "data-coco/annotations/json_annotation_test.json"
COCO_TEST_IMAGES = "data-coco/images/test"

SEM_DATA_SPLIT = 'data_sem_split.json'


# Detectron2 configuration

# pretrained model used at first
CONFIG_FILE = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"

# other pretrained model (slower)
# CONFIG_FILE = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"

OUTPUT_DIR = "retinaNet/output"
OUTPUT_TRUE_LABELS = "data-coco/images_true_label"
