import time
import wandb
from detectron2.engine import DefaultTrainer, hooks
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from constants.constants import COCO_VAL_ANNOTATION, COCO_VAL_IMAGES


class WandBTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.start_time = time.time()
        
    def run_step(self):
        super().run_step()
        
        # Get last registered metrics (loss, learning rate, etc)
        metrics_dict = {key: float(value[0]) for key, value in self.storage._latest_scalars.items()}

        for key, value in metrics_dict.items():
            wandb.log({key: value})
        
        training_time = time.time() - self.start_time
        wandb.log({"training_time": training_time})

    def evaluate(self):
        evaluator = COCOEvaluator(COCO_VAL_ANNOTATION, output_dir="./output/")
        val_loader = build_detection_test_loader(self.cfg, COCO_VAL_ANNOTATION)

        # FIXME: Error is thrown during inference
        results = inference_on_dataset(self.model, val_loader, evaluator)
        wandb.log(results)
        return results