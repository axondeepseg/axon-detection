import os
import wandb
import torch
from constants.data_constants import SEM, TEM
from retinaNet.trainer import Trainer
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor


from preprocessing import preprocess_data_coco
from utils import clear_directories_coco

from retinaNet.visualisations import visualize_true_labels

from retinaNet.constants.data_file_constants import (
    COCO_TEST_REG_NAME,
    COCO_TEST_SEM_ANNOTATION,
    COCO_TEST_SEM_IMAGES,
    COCO_TRAIN_REG_NAME,
    COCO_TRAIN_SEM_ANNOTATION,
    COCO_TRAIN_SEM_IMAGES,
    COCO_VAL_REG_NAME,
    COCO_VAL_SEM_ANNOTATION,
    COCO_VAL_SEM_IMAGES,
    COCO_VAL_TEM_ANNOTATION,
    COCO_VAL_TEM_IMAGES,
    COCO_TRAIN_TEM_ANNOTATION,
    COCO_TRAIN_TEM_IMAGES,
    COCO_TEST_TEM_ANNOTATION,
    COCO_TEST_TEM_IMAGES,
    COCO_VAL_TEM_ANNOTATION,
    SEM_DATA_SPLIT,
    CONFIG_FILE,
    OUTPUT_DIR,
)
from retinaNet.constants.wanb_config_constants import (
    WANDB_ENTITY,
    WANDB_PROJECT,
    WANDB_RUN_NAME,
)
from retinaNet.constants.config_constants import CONF_THRESHOLD

from detectron2.modeling import build_model
from detectron2.modeling.anchor_generator import build_anchor_generator

from detectron2.data import transforms as T

from detectron2.modeling import BACKBONE_REGISTRY, Backbone
from detectron2.layers import ShapeSpec
import torch
import timm
import torch.nn as nn


import numpy as np
from sklearn.cluster import KMeans
from detectron2.data import DatasetCatalog
from retinaNet.constants.data_file_constants import COCO_TRAIN_REG_NAME

@BACKBONE_REGISTRY.register()
class EfficientNetBackbone(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        self.model = timm.create_model("tf_efficientnet_b5", features_only=True, pretrained=True)
        
        self._out_features = ["0", "1", "2", "3", "4"]
        self._out_feature_channels = {"0": 24, "1": 40, "2": 64, "3": 176, "4": 512 }
        self._out_feature_strides = {"0": 4, "1": 8, "2": 16, "3": 32, "4": 64}

        self.proj_layers = nn.ModuleDict({
            name: nn.Conv2d(in_channels, 256, kernel_size=1) 
            for name, in_channels in self._out_feature_channels.items()
        })

    def forward(self, x):
        features = self.model(x)
        
        
        extra_feature = torch.nn.functional.adaptive_avg_pool2d(features[-1], output_size=(1, 1)) 
        features.append(extra_feature) 
        projected_features = {}
        
        for i, f in enumerate(features):
            feature_name = str(i)
            print(f"Applying Conv2d to Feature {feature_name} with shape {f.shape}")
            
            if feature_name not in self.proj_layers:
                print(f"Warning: {feature_name} is not in proj_layers.")
                continue
            
            projected_features[feature_name] = self.proj_layers[feature_name](f)
            
        return {str(i): self.proj_layers[str(i)](f) for i, f in enumerate(features)}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=256, 
                stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

def register_instances(data_type):

    if data_type == SEM:
        train_annotation = COCO_TRAIN_SEM_ANNOTATION
        val_annotation = COCO_VAL_SEM_ANNOTATION
        test_annotation = COCO_TEST_SEM_ANNOTATION
        train_images = COCO_TRAIN_SEM_IMAGES
        val_images = COCO_VAL_SEM_IMAGES
        test_images = COCO_TEST_SEM_IMAGES
    elif data_type == TEM:
        train_annotation = COCO_TRAIN_TEM_ANNOTATION
        val_annotation = COCO_VAL_TEM_ANNOTATION
        test_annotation = COCO_TEST_TEM_ANNOTATION
        train_images = COCO_TRAIN_TEM_IMAGES
        val_images = COCO_VAL_TEM_IMAGES
        test_images = COCO_TEST_TEM_IMAGES

    if (COCO_TRAIN_REG_NAME) not in list(MetadataCatalog):
        print("registered")
        register_coco_instances(COCO_TRAIN_REG_NAME, {}, train_annotation, train_images)

    if (COCO_VAL_REG_NAME) not in list(MetadataCatalog):
        register_coco_instances(COCO_VAL_REG_NAME, {}, val_annotation, val_images)

    if (COCO_TEST_REG_NAME) not in list(MetadataCatalog):
        register_coco_instances(COCO_TEST_REG_NAME, {}, test_annotation, test_images)

    print("List Meta")
    print(list(MetadataCatalog))
    print("get ids!!!")
    # print(MetadataCatalog.get(COCO_VAL_REGISTRATION).dataset_id_to_contiguous_id)
    print(MetadataCatalog.get(COCO_VAL_REG_NAME).get("thing_classes"))


def reset_instances():
    for annotation in [COCO_TRAIN_REG_NAME, COCO_VAL_REG_NAME, COCO_TEST_REG_NAME]:
        if annotation in list(MetadataCatalog):
            print("removed from metadata")
            MetadataCatalog.remove(annotation)

    for annotation in [COCO_TRAIN_REG_NAME, COCO_VAL_REG_NAME, COCO_TEST_REG_NAME]:
        if annotation in list(DatasetCatalog):
            DatasetCatalog.remove(annotation)

    print("METADATA CATALOG")
    print(list(MetadataCatalog))


def configure_detectron():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(CONFIG_FILE))
    cfg.DATASETS.TRAIN = (COCO_TRAIN_REG_NAME,)
    cfg.DATASETS.TEST = (COCO_TEST_REG_NAME,)
    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 350
    # cfg.SOLVER.STEPS = [40, 80]  # no learning decay (lr remains stable)
    # cfg.SOLVER.GAMMA = 0.1  # decay factor for lr
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"  # scheduler for early warmup
    cfg.SOLVER.WARMUP_ITERS = 50
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"

    print("\n -- solver")
    print(cfg.SOLVER)

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(CONFIG_FILE)
    
    # cfg.MODEL.BACKBONE.NAME = "EfficientNetBackbone"
    # cfg.MODEL.PIXEL_MEAN = [123.675, 116.28, 103.53] 
    # cfg.MODEL.PIXEL_STD = [58.395, 57.12, 57.375]
    # cfg.MODEL.RETINANET.IN_FEATURES = ["0", "1", "2", "3", "4"]

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    
    # Force Detectron2 to use GPU
    cfg.MODEL.DEVICE = "cuda"
    print("Using device:", cfg.MODEL.DEVICE)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but GPU mode was requested!")

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONF_THRESHOLD
    cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA = 1
    cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA = 0.5
    
    cfg.MODEL.RETINANET.BBOX_REG_LOSS_TYPE = "smooth_l1"
    
    # cfg.INPUT.RANDOM_FLIP = "horizontal"
    
    cfg.INPUT.AUGMENTATIONS = [
        T.RandomResize([800, 1200]),
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        T.RandomSaturation(1, 1.2),
        T.RandomLighting(0.7),
    ]

    # TODO: Find right anchor boxes

    def get_bbox_sizes(dataset_name):
        """Extracts bounding box widths and heights from the dataset."""
        dataset_dicts = DatasetCatalog.get(dataset_name)
        bbox_sizes = []

        for data in dataset_dicts:
            for annotation in data["annotations"]:
                x, y, w, h = annotation["bbox"]  # COCO format: [x_min, y_min, width, height]
                bbox_sizes.append([w, h])

        return np.array(bbox_sizes)

    def kmeans_anchors(bbox_sizes, num_clusters):
        """Runs K-Means clustering to find optimal anchor sizes."""
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        kmeans.fit(bbox_sizes)
        
        return np.sort(kmeans.cluster_centers_, axis=0)  # Sort anchors by size

    # Extract bounding box sizes from dataset
    bbox_sizes = get_bbox_sizes(COCO_TRAIN_REG_NAME)

    # Optimize anchors using K-Means
    num_anchors = 5  # Choose based on your model needs
    optimized_anchors = kmeans_anchors(bbox_sizes, num_anchors)
    
    optimized_anchors = np.array(optimized_anchors).reshape((5, 1, 2))
    optimized_anchors = optimized_anchors[:, 0, :]
    optimized_anchor_sizes = [[size[0], size[1], size[0] * 1.2] for size in optimized_anchors]

    print(optimized_anchors.shape)

    print("Optimized Anchor Sizes:", optimized_anchor_sizes)

    # Update Detectron2 configuration
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = optimized_anchor_sizes


    print("\n -- model")
    print(cfg.MODEL)

    cfg.OUTPUT_DIR = OUTPUT_DIR
    cfg.TEST.DETECTIONS_PER_IMAGE = 2000

    return cfg


def clear_data():
    split_file = SEM_DATA_SPLIT
    if os.path.exists(split_file):
        os.remove(split_file)
        print(f"{split_file} has been deleted.")
    else:
        print(f"{split_file} does not exist.")

    clear_directories_coco()


if __name__ == "__main__":
    

    # TODO: Run this only once when the registered metadata isnt the same as local

    # clear_data()
    # preprocess_data_coco(TEM)
    # visualize_true_labels(COCO_TEST_TEM_ANNOTATION, data_type=TEM, set_type="test")

    # TRAIN STEPS:

    setup_logger()
    reset_instances()

    register_instances(TEM)
    cfg = configure_detectron()

    api = wandb.Api()

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        dir="/output",
        # mode="offline",
    )

    run.config.update(
        {
            "train_dataset": cfg.DATASETS.TRAIN,
            "test_dataset": cfg.DATASETS.TEST,
            "num_workers": cfg.DATALOADER.NUM_WORKERS,
            "batch_size": cfg.SOLVER.IMS_PER_BATCH,
            "base_lr": cfg.SOLVER.BASE_LR,
            "max_iter": cfg.SOLVER.MAX_ITER,
            "lr_steps": cfg.SOLVER.STEPS,
            "lr_scheduler": cfg.SOLVER.LR_SCHEDULER_NAME,
            "warmup_iters": cfg.SOLVER.WARMUP_ITERS,
            "clip_gradients_enabled": cfg.SOLVER.CLIP_GRADIENTS.ENABLED,
            "clip_type": cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE,
            "weights": cfg.MODEL.WEIGHTS,
            "roi_heads_batch_size": cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "device": cfg.MODEL.DEVICE,
            "score_threshold": cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "focal_loss_alpha": cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA,
            "focal_loss_gamma": cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA,
            "anchor_aspect_ratio": cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS,
            "output_dir": cfg.OUTPUT_DIR,
            "detections_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
        }
    )

    # TODO: remove when training
    # cfg.MODEL.WEIGHTS = "retinaNet/output/model_final.pth"

    model_trainer = Trainer(cfg)

    # TODO: Add when training
    model_trainer.resume_or_load(resume=False)

    try:
        model_trainer.train()
    except Exception as e:
        print("Training stopped due to:" + str(e))

    try:
        final_test_metrics = model_trainer.test(model_trainer.cfg, model_trainer.model)
        run.log(final_test_metrics)
    except Exception as e:
        print("Validation run stopped due to:" + str(e))

    model_path = "retinaNet/output/model_final.pth"
    torch.save(model_trainer.model.state_dict(), model_path)

    model_trainer.visualize_predictions(COCO_TEST_TEM_IMAGES)
