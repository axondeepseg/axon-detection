import itertools
import wandb
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator
from constants.data_constants import TEM
from retinaNet.constants.data_file_constants import COCO_TEST_REG_NAME
from retinaNet.retinaNet_train import configure_detectron, reset_instances, register_instances

from retinaNet.constants.wanb_config_constants import WANDB_ENTITY, WANDB_PARAM_SEARCH, WANDB_RUN_NAME
from retinaNet.trainer import Trainer

def get_train_cfg(search_space):
    cfg = configure_detectron()
    
    for key, value in search_space:
        cfg["RETINANET"][key] = value
        
    cfg.OUTPUT_DIR = "./output/"
    return cfg

def train_and_evaluate(search_space):
    cfg = get_train_cfg(search_space)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    evaluator = COCOEvaluator(COCO_TEST_REG_NAME, cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_results = trainer.test(cfg, trainer.model)
    return val_results

def hyperparameter_search(search_space):
    
    keys, values = zip(*search_space.items())
    for v in itertools.product(*values):
        params = dict(zip(keys, v))
        print(f"Training with params: {params}")
        
        run_name = f"RUN_BBOX_LOSS:{params["BBOX_REG_LOSS_TYPE"]}_FL_GAMMA:{params["FOCAL_LOSS_GAMMA"]}"
        wandb.init(entity=WANDB_ENTITY, project=WANDB_PARAM_SEARCH, name=run_name, reinit=True)
        results = train_and_evaluate(params)
        
        wandb.log({"params": params, **results})
        wandb.finish()
    
if __name__ == "__main__":
    reset_instances()
    register_instances(TEM)
    
    search_space = {
        "max_iter": [250],
        "BBOX_REG_LOSS_TYPE": ["smooth_l1", "giou"],
        "FOCAL_LOSS_GAMMA": [2, 5, 6, 8]
    }
    hyperparameter_search(search_space)
