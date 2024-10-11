import time
import wandb
from detectron2.engine import DefaultTrainer, hooks
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

class WandBTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.start_time = time.time()
        
    def run_step(self):
        super().run_step()
        
        # Get latest metrics (loss, learning rate, etc.) from the storage
        metrics_dict = self.storage._latest_scalars
        
        wandb.log(metrics_dict)
        
        # Log training time (relative to start)
        training_time = time.time() - self.start_time
        wandb.log({"training_time": training_time})

    def evaluate(self):
        evaluator = COCOEvaluator("your_dataset_name", self.cfg, False, output_dir="./output/")
        val_loader = build_detection_test_loader(self.cfg, "your_dataset_name")
        results = inference_on_dataset(self.model, val_loader, evaluator)
        wandb.log(results)
        return results