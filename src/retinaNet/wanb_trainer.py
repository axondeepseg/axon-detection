import time
import wandb as wandb
import matplotlib as plt
from detectron2.engine import DefaultTrainer, hooks
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from retinaNet.constants.constants import COCO_TEST_ANNOTATION, COCO_VAL_ANNOTATION, COCO_VAL_IMAGES


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


    def log_metrics(self, results, split_name="Validation"):
        """
        Log evaluation metrics to WandB.
        """
        metrics = {
            f"{split_name}_mAP": results["bbox"]["AP"],   # mean Average Precision
            f"{split_name}_AP50": results["bbox"]["AP50"], # AP at IoU=50
            f"{split_name}_AP75": results["bbox"]["AP75"], # AP at IoU=75
            f"{split_name}_AR": results["bbox"]["AR"],     # Average Recall
        }
        
        # Calculate F1 Score based on AP and AR
        precision = results["bbox"]["AP"]  # Using AP as a proxy for precision
        recall = results["bbox"]["AR"]     # Using AR as a proxy for recall
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        metrics[f"{split_name}_F1"] = f1_score
        
        wandb.log(metrics)

        # Plot F1 score as a separate line plot
        plt.plot(f1_score, marker="o", label=f"{split_name}_F1 Score")
        plt.title(f"{split_name} F1 Score Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("F1 Score")
        plt.legend()
        wandb.log({f"{split_name}_F1_Score_Plot": wandb.Image(plt)})
        plt.clf()  # Clear figure after logging


    def evaluate(self):
        evaluator = COCOEvaluator(COCO_VAL_ANNOTATION, output_dir="./output/")
        val_loader = build_detection_test_loader(self.cfg, COCO_VAL_ANNOTATION)

        # FIXME: Error is thrown during inference
        results = inference_on_dataset(self.model, val_loader, evaluator)
        wandb.log(results)
        return results
    
    def test(self):
        test_evaluator = COCOEvaluator(COCO_TEST_ANNOTATION, output_dir="./output/")
        test_loader = build_detection_test_loader(self.cfg, COCO_TEST_ANNOTATION)

        test_results = inference_on_dataset(self.model, test_loader, test_evaluator)
        wandb.log(test_results)
        return test_results 