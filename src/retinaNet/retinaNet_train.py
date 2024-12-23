import os
import wandb
from retinaNet.wanb_trainer import WandBTrainer
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog
from preprocessing import preprocess_data_coco
from utils import clear_directories_coco

from retinaNet.constants.constants import SEM_DATA_SPLIT, COCO_TRAIN_ANNOTATION, COCO_TRAIN_IMAGES, COCO_VAL_ANNOTATION, COCO_VAL_IMAGES, COCO_TEST_ANNOTATION, COCO_TEST_IMAGES, CONFIG_FILE, OUTPUT_DIR
from retinaNet.constants.wanb_config_constants import WANDB_ENTITY, WANDB_PROJECT, WANDB_RUN_NAME, WANDB_RUN_ID

def register_instances():

    if (COCO_TRAIN_ANNOTATION, COCO_VAL_IMAGES) not in list(MetadataCatalog):
        register_coco_instances(COCO_TRAIN_ANNOTATION, {}, COCO_TRAIN_ANNOTATION, COCO_TRAIN_IMAGES)

    if (COCO_VAL_ANNOTATION, COCO_VAL_IMAGES) not in list(MetadataCatalog):
        register_coco_instances(COCO_VAL_ANNOTATION, {}, COCO_VAL_ANNOTATION, COCO_VAL_IMAGES)

    if (COCO_TEST_ANNOTATION, COCO_TEST_IMAGES) not in list(MetadataCatalog):
        register_coco_instances(COCO_TEST_ANNOTATION, {}, COCO_TEST_ANNOTATION, COCO_TEST_IMAGES)

        # FIXME: These lines don't work since thing_classes already has a value (axon and myelin) and thing_dataset_id_to_contiguous_id
        # MetadataCatalog.get(COCO_VAL_ANNOTATION).set(thing_classes=["axon"])
        # thing_classes = MetadataCatalog.get(COCO_VAL_ANNOTATION).thing_classes
        # dataset_id_contiguous_id = {i: i for i in range(len(["axon", "myelin"]))}
        # MetadataCatalog.get(COCO_VAL_ANNOTATION).set(thing_dataset_id_to_contiguous_id=dataset_id_contiguous_id)


def reset_instance():
    MetadataCatalog.remove(COCO_VAL_ANNOTATION)

def configure_detectron():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(CONFIG_FILE))
    cfg.DATASETS.TRAIN = (COCO_TRAIN_ANNOTATION,)
    cfg.DATASETS.TEST = (COCO_TEST_ANNOTATION,)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(CONFIG_FILE)  
    cfg.SOLVER.IMS_PER_BATCH = 2  
    cfg.SOLVER.BASE_LR = 0.00025  
    cfg.SOLVER.MAX_ITER = 10
    cfg.SOLVER.STEPS = [] # no learning decay (lr remains stable)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
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


if __name__ == '__main__':

    clear_data()
    preprocess_data_coco()

    setup_logger()
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
    #     run.log(final_metrics)
    # except Exception as e:
    #     print('Validation run stopped due to:' + str(e))
