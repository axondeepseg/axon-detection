# Paths for COCO dataset
COCO_TRAIN_SEM_ANNOTATION = "data-coco/sem/annotations/json_annotation_train.json"
COCO_TRAIN_SEM_IMAGES = "data-coco/sem/images/train"
COCO_VAL_SEM_ANNOTATION = "data-coco/sem/annotations/json_annotation_val.json"
COCO_VAL_SEM_IMAGES = "data-coco/sem/images/val"
COCO_TEST_SEM_ANNOTATION = "data-coco/sem/annotations/json_annotation_test.json"
COCO_TEST_SEM_IMAGES = "data-coco/sem/images/test"

COCO_TRAIN_REG_NAME = "axon_detection_annotation_train"
COCO_VAL_REG_NAME = "axon_detection_annotation_val"
COCO_TEST_REG_NAME = "axon_detection_annotation_test"


# Paths for COCO TEM dataset
COCO_TRAIN_TEM_ANNOTATION = "data-coco/tem/annotations/json_annotation_train.json"
COCO_TRAIN_TEM_IMAGES = "data-coco/tem/images/train"
COCO_VAL_TEM_ANNOTATION = "data-coco/tem/annotations/json_annotation_val.json"
COCO_VAL_TEM_IMAGES = "data-coco/tem/images/val"
COCO_TEST_TEM_ANNOTATION = "data-coco/tem/annotations/json_annotation_test.json"
COCO_TEST_TEM_IMAGES = "data-coco/tem/images/test"


SEM_DATA_SPLIT = "data_sem_split.json"

# Detectron2 configuration

# pretrained model used at first
# CONFIG_FILE = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"

# other pretrained model (slower, but more performant)
CONFIG_FILE = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"

OUTPUT_DIR = "retinaNet/output"
OUTPUT_TRUE_LABELS = "data-coco/images_true_label"
