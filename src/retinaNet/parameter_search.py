import itertools
import wandb
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator
from detectron2.model_zoo import get_config_file
from constants.data_constants import TEM
from retinaNet.constants.data_file_constants import COCO_TEST_REG_NAME
from retinaNet.retinaNet_train import configure_detectron, reset_instances, register_instances

from retinaNet.constants.wanb_config_constants import WANDB_ENTITY, WANDB_PARAM_SEARCH, WANDB_RUN_NAME

def get_train_cfg(config_file, num_classes, base_lr, ims_per_batch, warmup_iters, max_iter):
    cfg = configure_detectron()
    cfg.MODEL.RETINANET.NUM_CLASSES = num_classes
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
    cfg.SOLVER.WARMUP_ITERS = warmup_iters
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.OUTPUT_DIR = "./output/"
    return cfg

def train_and_evaluate(config_file, num_classes, base_lr, ims_per_batch, warmup_iters, max_iter):
    cfg = get_train_cfg(config_file, num_classes, base_lr, ims_per_batch, warmup_iters, max_iter)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    evaluator = COCOEvaluator(COCO_TEST_REG_NAME, cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_results = trainer.test(cfg, trainer.model, evaluators=[evaluator])
    return val_results

def hyperparameter_search(config_file, num_classes, search_space):
    
    keys, values = zip(*search_space.items())
    for v in itertools.product(*values):
        params = dict(zip(keys, v))
        print(f"Training with params: {params}")
        
        run_name = f"RUN_LR-{params["base_lr"]}_BATCH-{params["ims_per_batch"]}_WARMUP-{params["warmup_iters"]}"
        wandb.init(entity=WANDB_ENTITY, project=WANDB_PARAM_SEARCH, name=run_name)
        results = train_and_evaluate(config_file, num_classes, **params)
        
        wandb.log({"params": params, **results})
    
if __name__ == "__main__":
    reset_instances()
    register_instances(TEM)
    
    search_space = {
        "base_lr": [0.001, 0.0005],
        "ims_per_batch": [1, 2],
        "warmup_iters": [20, 40],
        "max_iter": [350]
    }
    hyperparameter_search("COCO-Detection/retinanet_R_50_FPN_3x.yaml", 2, search_space)
