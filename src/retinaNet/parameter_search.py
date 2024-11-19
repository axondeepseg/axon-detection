import wandb
from detectron2.config import get_cfg
from detectron2 import model_zoo

from retinaNet.constants.config_constants import CONF_THRESHOLD
from retinaNet.constants.data_file_constants import COCO_TEST_ANNOTATION, COCO_TRAIN_ANNOTATION, CONFIG_FILE, OUTPUT_DIR
from retinaNet.constants.wanb_config_constants import WANDB_ENTITY
from retinaNet.trainer import Trainer

from retinaNet.retinaNet_train import register_instances

def configure_detectron_sweep(sweep_config):
    """
    Create detectron config based on sweep parameters
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(CONFIG_FILE))
    cfg.DATASETS.TRAIN = (COCO_TRAIN_ANNOTATION,)
    cfg.DATASETS.TEST = (COCO_TEST_ANNOTATION,)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(CONFIG_FILE)  

    # Update with sweep parameters
    cfg.SOLVER.BASE_LR = sweep_config["learning_rate"]
    cfg.SOLVER.IMS_PER_BATCH = sweep_config["batch_size"]
    cfg.SOLVER.MAX_ITER = 60
    cfg.SOLVER.STEPS = []  # Learning rate remains stable by default
    # cfg.SOLVER.GAMMA = 1.0
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = sweep_config["roi_batch_size"]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.DEVICE = "cpu"
    cfg.OUTPUT_DIR = OUTPUT_DIR
    cfg.TEST.DETECTIONS_PER_IMAGE = 2000
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONF_THRESHOLD

    return cfg


def train():
    """
    Main function to train the model for the sweep.
    """
    with wandb.init() as run:
        sweep_config = wandb.config

        parameters = {
            "Parameters/learning_rate": wandb.config.learning_rate,
            "Parameters/batch_size": wandb.config.batch_size,
            "Parameters/max_iter": wandb.config.max_iter,
            "Parameters/roi_batch_size": wandb.config.roi_batch_size
        }
        
        # Log hyperparameters explicitly
        wandb.config.update(parameters)
        wandb.log(parameters)
        
        cfg = configure_detectron_sweep(sweep_config)

        trainer = Trainer(cfg)
        trainer.resume_or_load(resume=False)

        try:
            trainer.train()
        except Exception as e:
            print(f"Training or evaluation stopped due to: {e}")

        evaluation_results = trainer.evaluate()
        mAP = evaluation_results["bbox"]["AP"]
        wandb.log({"Validation/Average_Precision": evaluation_results})

        wandb.log({"mAP": mAP})
        print('\nWANDB mAP score')
        print(mAP)


if __name__ == "__main__":

    register_instances()

    sweep_config = {
        "method": "grid",
        "parameters": {
            "learning_rate": {"values": [0.0001, 0.00025, 0.001]},
            "batch_size": {"values": [2]},
            "max_iter": {"values": [60]},
            "roi_batch_size": {"values": [128, 256]},
            "learning_rate_decay": {"values": [1.0, 0.01]},
        },
    }

    sweep_id = wandb.sweep(sweep_config, project="retinanet-parameter_search")
    wandb.agent(sweep_id, function=train)
