import os
import wandb
import cv2
import glob
from retinaNet.wanb_trainer import WandBTrainer
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor

from preprocessing import preprocess_data_coco
from utils import clear_directories_coco

from retinaNet.constants.constants import SEM_DATA_SPLIT, COCO_TRAIN_ANNOTATION, COCO_TRAIN_IMAGES, COCO_VAL_ANNOTATION, COCO_VAL_IMAGES, COCO_TEST_ANNOTATION, COCO_TEST_IMAGES, CONFIG_FILE, OUTPUT_DIR
from retinaNet.constants.wanb_config_constants import WANDB_ENTITY, WANDB_PROJECT, WANDB_RUN_NAME, WANDB_RUN_ID

def register_instances():
    print(list(MetadataCatalog))
    print('\n data set:\n')
    print(list(DatasetCatalog))

    # MetadataCatalog.get(COCO_TRAIN_ANNOTATION).set(thing_classes=["axon"])
    # MetadataCatalog.get(COCO_VAL_ANNOTATION).set(thing_classes=["axon"])
    # MetadataCatalog.get(COCO_TRAIN_ANNOTATION).set(thing_dataset_id_to_contiguous_id={0: 0})
    # MetadataCatalog.get(COCO_VAL_ANNOTATION).set(thing_dataset_id_to_contiguous_id={0: 0})

    if (COCO_TRAIN_ANNOTATION, COCO_VAL_IMAGES) not in list(MetadataCatalog):
        register_coco_instances(COCO_TRAIN_ANNOTATION, {}, COCO_TRAIN_ANNOTATION, COCO_TRAIN_IMAGES)

    if (COCO_VAL_ANNOTATION, COCO_VAL_IMAGES) not in list(MetadataCatalog):
        register_coco_instances(COCO_VAL_ANNOTATION, {}, COCO_VAL_ANNOTATION, COCO_VAL_IMAGES)

    if (COCO_TEST_ANNOTATION, COCO_TEST_IMAGES) not in list(MetadataCatalog):
        register_coco_instances(COCO_TEST_ANNOTATION, {}, COCO_TEST_ANNOTATION, COCO_TEST_IMAGES)

    print('\nAFTER\n')
    print(list(MetadataCatalog))

        # FIXME: These lines don't work since thing_classes already has a value (axon and myelin) and thing_dataset_id_to_contiguous_id
        # thing_classes = MetadataCatalog.get(COCO_VAL_ANNOTATION).thing_classes
        # dataset_id_contiguous_id = {i: i for i in range(len(["axon", "myelin"]))}
        # MetadataCatalog.get(COCO_VAL_ANNOTATION).set(thing_dataset_id_to_contiguous_id=dataset_id_contiguous_id)


def reset_instances():
    for annotation in [COCO_TRAIN_ANNOTATION, COCO_VAL_ANNOTATION, COCO_TEST_ANNOTATION]:
        if annotation in list(MetadataCatalog):
            MetadataCatalog.remove(annotation)

    for annotation in [COCO_TRAIN_ANNOTATION, COCO_VAL_ANNOTATION, COCO_TEST_ANNOTATION]:
        if annotation in list(DatasetCatalog):
            DatasetCatalog.remove(annotation)

    print(list(MetadataCatalog))
    print('\n meta data set:\n')
    print(MetadataCatalog.get(COCO_TRAIN_ANNOTATION))
    print(MetadataCatalog.get(COCO_VAL_ANNOTATION))



def configure_detectron():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(CONFIG_FILE))
    cfg.DATASETS.TRAIN = (COCO_TRAIN_ANNOTATION,)
    cfg.DATASETS.TEST = (COCO_TEST_ANNOTATION,)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(CONFIG_FILE)  
    cfg.SOLVER.IMS_PER_BATCH = 2  
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 80
    # cfg.SOLVER.STEPS = [] # no learning decay (lr remains stable)
    # cfg.SOLVER.STEPS = [20, 30] # change according to max iter
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR" # constant decay
    cfg.SOLVER.GAMMA = 0.1  # decay factor for lr
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.DEVICE = "cpu"  
    cfg.OUTPUT_DIR = OUTPUT_DIR

    return cfg

def clear_data():
    split_file = SEM_DATA_SPLIT
    if os.path.exists(split_file):
        os.remove(split_file)
        print(f"{split_file} has been deleted.")
    else:
        print(f"{split_file} does not exist.")
    
    clear_directories_coco()


# def visualize_predictions(cfg, test_dir=COCO_TEST_IMAGES, output_dir="output_predictions", conf_threshold=0.5):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_threshold
#     predictor = DefaultPredictor(cfg)
#     image_paths = glob.glob(os.path.join(test_dir, "*.png"))

#     for image_path in image_paths:
#         print(f"Predicting on image: {image_path}")
#         img = cv2.imread(image_path)

#         # Make prediction
#         outputs = predictor(img)
#         instances = outputs["instances"].to("cpu")

#         scores = instances.scores.numpy()
#         boxes = instances.pred_boxes.tensor.numpy()
#         classes = instances.pred_classes.numpy()

#         # Filter out predictions below the confidence threshold
#         filtered_boxes = [box for i, box in enumerate(boxes) if scores[i] >= conf_threshold]
#         filtered_scores = [score for i, score in enumerate(scores) if score >= conf_threshold]
#         filtered_classes = [cls for i, cls in enumerate(classes) if scores[i] >= conf_threshold]

#         # Draw bounding boxes and confidence scores
#         for i, box in enumerate(filtered_boxes):
#             x1, y1, x2, y2 = map(int, box)
#             # confidence = filtered_scores[i]
#             class_id = filtered_classes[i]

#             # Set color based on class id
#             color = (0, 255, 0) if class_id == 0 else (255, 0, 0)  # Green for 'axon', Blue for 'myelin'
            
#             # Draw rectangle and confidence
#             cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)  

#         # Save and log image
#         output_path = os.path.join(output_dir, os.path.basename(image_path))
#         success = cv2.imwrite(output_path, img)
#         if success:
#             print(f"Saved prediction image to {output_path}")
#         else:
#             print(f"Failed to save prediction image to {output_path}")

#         wandb.log({"Prediction": wandb.Image(img, caption=os.path.basename(image_path))})

#     print("Predictions visualized and logged to WandB.")

def visualize_predictions(cfg, test_dir, conf=0.25):
        output_directory = 'output_predictions'
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf
        predictor = DefaultPredictor(cfg)
        image_paths = glob.glob(os.path.join(test_dir, "*.png"))
        
        for image_path in image_paths:
            img = cv2.imread(image_path)
            print("Predicting on image:", image_path)
            outputs = predictor(img)
            instances = outputs["instances"].to("cpu")

            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()

            print('boxes')
            print(len(boxes))
            print(boxes)

            # Filter out predictions below the confidence threshold
            filtered_boxes = [box for i, box in enumerate(boxes)]

            for i, box in enumerate(filtered_boxes):
                print('filtered boxes sizes')
                print(len(filtered_boxes))
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            output_path = os.path.join(output_directory, os.path.basename(image_path))
            success = cv2.imwrite(output_path, img)
            if success:
                print(f"Saved prediction image to {output_path}")
            else:
                print(f"Failed to save prediction image to {output_path}")
                
            # wandb.log({"Prediction": [wandb.Image(img, caption=os.path.basename(image_path))]})

if __name__ == '__main__':

    # clear_data()
    # preprocess_data_coco()

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
        id=WANDB_RUN_ID,
        dir='/output'
    )

    run.config.update({
        "learning_rate": cfg.SOLVER.BASE_LR,
        "batch_size": cfg.SOLVER.IMS_PER_BATCH,
        "max_iter": cfg.SOLVER.MAX_ITER
    })

    trainer = WandBTrainer(cfg)
    trainer.resume_or_load(resume=False)

    try:
        trainer.train()
    except Exception as e:
        print('Training stopped due to:' + str(e))

    # TODO: Fix evaluation method
    # try:
    #     final_val_metrics = trainer.evaluate()
    #     run.log(final_val_metrics)
    # except Exception as e:
    #     print('Validation run stopped due to:' + str(e))

    # try:
    #     final_test_metrics = trainer.test()
    #     run.log(final_test_metrics)
    # except Exception as e:
    #     print('Validation run stopped due to:' + str(e))

    cfg.MODEL.WEIGHTS = "retinaNet/output/model_final.pth"  
    # cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.25

    visualize_predictions(cfg, COCO_TEST_IMAGES)
