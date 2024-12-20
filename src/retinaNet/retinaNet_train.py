import os
import time
import wandb
import cv2
import glob
from retinaNet.trainer import Trainer

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor

from preprocessing import preprocess_data_coco
from utils import clear_directories_coco

from retinaNet.visualisations import visualize_true_labels

from retinaNet.constants.data_file_constants import (
    COCO_TEST_REG_NAME,
    COCO_TRAIN_REG_NAME,
    COCO_VAL_REG_NAME,
    # COCO_VAL_TEM_ANNOTATION,
    # COCO_VAL_TEM_IMAGES,
    # COCO_TRAIN_TEM_ANNOTATION,
    # COCO_TRAIN_TEM_IMAGES,
    # COCO_TEST_TEM_ANNOTATION,
    # COCO_TEST_TEM_IMAGES,
    COCO_VAL_SEM_ANNOTATION,
    COCO_VAL_SEM_IMAGES,
    COCO_TRAIN_SEM_ANNOTATION,
    COCO_TRAIN_SEM_IMAGES,
    COCO_TEST_SEM_ANNOTATION,
    COCO_TEST_SEM_IMAGES,
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


def register_instances(data_type: str = ""):

    if (COCO_TRAIN_REG_NAME) not in list(MetadataCatalog):
        print("registered")
        register_coco_instances(
            COCO_TRAIN_REG_NAME, {}, COCO_TRAIN_SEM_ANNOTATION, COCO_TRAIN_SEM_IMAGES
        )

    if (COCO_VAL_REG_NAME) not in list(MetadataCatalog):
        register_coco_instances(
            COCO_VAL_REG_NAME, {}, COCO_VAL_SEM_ANNOTATION, COCO_VAL_SEM_IMAGES
        )

    if (COCO_TEST_REG_NAME) not in list(MetadataCatalog):
        register_coco_instances(
            COCO_TEST_REG_NAME, {}, COCO_TEST_SEM_ANNOTATION, COCO_TEST_SEM_IMAGES
        )

    print("List Meta")
    print(list(MetadataCatalog))
    thing_classes = ["axon", "not-axon"]
    print("get ids!!!")
    # print(MetadataCatalog.get(COCO_VAL_REGISTRATION).dataset_id_to_contiguous_id)
    print(MetadataCatalog.get(COCO_VAL_REG_NAME).get("thing_classes"))

    # FIXME: These lines don't work since thing_classes already has a value (axon and myelin) and thing_dataset_id_to_contiguous_id
    # print("settings things id")
    # dataset_id_contiguous_id = {1: 1}
    # MetadataCatalog.get(COCO).set(
    #     thing_dataset_id_to_contiguous_id=dataset_id_contiguous_id
    # )


def reset_instances():
    for annotation in [COCO_TRAIN_REG_NAME, COCO_VAL_REG_NAME, COCO_TEST_REG_NAME]:
        if annotation in list(MetadataCatalog):
            print("removed from metadata")
            MetadataCatalog.remove(annotation)

    for annotation in [COCO_TRAIN_REG_NAME, COCO_VAL_REG_NAME, COCO_TEST_REG_NAME]:
        if annotation in list(DatasetCatalog):
            DatasetCatalog.remove(annotation)

    print(list(MetadataCatalog))
    # print('\n meta data set:\n')
    # print(MetadataCatalog.get(COCO_TRAIN_ANNOTATION))
    # print(MetadataCatalog.get(COCO_VAL_ANNOTATION))


def configure_detectron():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(CONFIG_FILE))
    cfg.DATASETS.TRAIN = (COCO_TRAIN_REG_NAME,)
    cfg.DATASETS.TEST = (COCO_TEST_REG_NAME,)
    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 200  # (2*140)/8 = 60 epochs
    cfg.SOLVER.STEPS = [80, 120]  # no learning decay (lr remains stable)
    # cfg.SOLVER.GAMMA = 0.1  # decay factor for lr
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"  # scheduler for early warmup
    cfg.SOLVER.WARMUP_ITERS = 40
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"

    print("\n -- solver")
    print(cfg.SOLVER)

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(CONFIG_FILE)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONF_THRESHOLD
    cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA = 5
    cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA = 0.5
    cfg.MODEL.RETINANET.NUM_CLASSES = 1

    # TODO: Find right anchor boxes through script
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [16, 32, 64, 128, 256]

    # No shown improvement
    # cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.412, 1.0, 2.43]]

    print("\n -- model")
    print(cfg.MODEL)

    cfg.OUTPUT_DIR = OUTPUT_DIR
    cfg.TEST.DETECTIONS_PER_IMAGE = 2000

    return cfg


def clear_data():
    # split_file = SEM_DATA_SPLIT
    # if os.path.exists(split_file):
    #     os.remove(split_file)
    #     print(f"{split_file} has been deleted.")
    # else:
    #     print(f"{split_file} does not exist.")

    clear_directories_coco()


def visualize_predictions(cfg, test_dir, conf_threshold=CONF_THRESHOLD):
    output_directory = "output_predictions"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_threshold
    predictor = DefaultPredictor(cfg)
    image_paths = glob.glob(os.path.join(test_dir, "*.png"))

    for image_path in image_paths:
        img = cv2.imread(image_path)
        print("Predicting on image:", image_path)

        start_time = time.time()
        outputs = predictor(img)
        inference_time = time.time() - start_time

        wandb.log({"Inference Time (s)": inference_time})
        print(f"Inference time for {image_path}: {inference_time:.4f} seconds")

        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()

        print("boxes")
        print(len(boxes))
        print(boxes)

        for i, box in enumerate(boxes):
            if scores[i] > conf_threshold:
                x1, y1, width, height = map(int, box)
                x2 = x1 + width
                y2 = y1 + height
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        output_path = os.path.join(output_directory, os.path.basename(image_path))
        success = cv2.imwrite(output_path, img)
        if success:
            print(f"Saved prediction image to {output_path}")
        else:
            print(f"Failed to save prediction image to {output_path}")

        wandb.log(
            {"Prediction": [wandb.Image(img, caption=os.path.basename(image_path))]}
        )


if __name__ == "__main__":

    # # TODO: Run this only once when the registered metadata isnt the same as local
    # clear_data()
    # preprocess_data_coco("data_axondeepseg_sem")
    visualize_true_labels(COCO_VAL_SEM_ANNOTATION, data_type="sem", set_type="val")
    # visualize_true_labels(COCO_TEST_IMAGES, set_type="test")

    # TRAIN STEPS:

    setup_logger()
    reset_instances()
    register_instances()
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
    model_trainer.resume_or_load(resume=False)

    try:
        print("\n-- TRAIN")
        model_trainer.train()
    except Exception as e:
        print("Training stopped due to:" + str(e))

    try:
        print("\n-- VAL")
        final_val_metrics = model_trainer.evaluate()
        run.log(final_val_metrics)
    except Exception as e:
        print("Validation run stopped due to:" + str(e))

    # try:
    #     final_test_metrics = model_trainer.test()
    #     run.log(final_test_metrics)
    # except Exception as e:
    #     print("Validation run stopped due to:" + str(e))

    # cfg.MODEL.WEIGHTS = "retinaNet/output/model_final.pth"

    visualize_predictions(cfg, COCO_TEST_SEM_IMAGES)
    visualize_true_labels(COCO_TEST_SEM_ANNOTATION, set_type="test")
