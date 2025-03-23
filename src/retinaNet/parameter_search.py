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
        
    cfg["SOLVER"]["MAX_ITER"] = search_space.get("MAX_ITER")
    cfg["MODEL"]["RETINANET"]["FOCAL_LOSS_GAMMA"] = search_space.get("FOCAL_LOSS_GAMMA")
    cfg["MODEL"]["RETINANET"]["BBOX_REG_LOSS_TYPE"] = search_space.get("BBOX_REG_LOSS_TYPE")
    cfg["MODEL"]["RETINANET"]["FOCAL_LOSS_ALPHA"] = search_space.get("FOCAL_LOSS_ALPHA")
        
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
        
        run_name = f"FL_GAMMA:{params["FOCAL_LOSS_GAMMA"]}"
        wandb.init(entity=WANDB_ENTITY, project=WANDB_PARAM_SEARCH, name=run_name, reinit=True)
        results = train_and_evaluate(params)
        
        wandb.log({"params": params, **results})
        wandb.finish()
    
if __name__ == "__main__":
    reset_instances()
    register_instances(TEM)
    
    search_space = {
        "MAX_ITER": [250],
        "BBOX_REG_LOSS_TYPE": ["smooth_l1"],
        "FOCAL_LOSS_GAMMA": [1], 
        "FOCAL_LOSS_ALPHA": [0.5],
    }
    hyperparameter_search(search_space)
