import glob
import os
import time
import cv2
import wandb as wandb
import matplotlib.pyplot as plt
import torch
from detectron2.engine import DefaultTrainer, hooks
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from retinaNet.constants.constants import COCO_TEST_ANNOTATION, COCO_VAL_ANNOTATION, COCO_VAL_IMAGES

from detectron2.engine import DefaultPredictor


class WandBTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.start_time = time.time()
        self.predictor = DefaultPredictor(self.cfg)
        
    def run_step(self):
        super().run_step()
        
        metrics_dict = {key: float(value[0]) for key, value in self.storage._latest_scalars.items()}
        
        # Plot all registered metrics (loss, learning rate, etc)
        current_iteration = self.storage.iter
        for key, value in metrics_dict.items():
            wandb.log({key: value})
        
        current_lr = self.optimizer.param_groups[0]["lr"]
        wandb.log({"learning_rate": current_lr})
        
        training_time = time.time() - self.start_time
        wandb.log({"training_time": training_time})


        # PREDICTION part for visualization of result

        current_lr = self.optimizer.param_groups[0]["lr"]
        print(f"\nLearning rate at epoch {current_iteration}: {current_lr}")

        self.predictor.model.load_state_dict(self.model.state_dict())

        image_paths = glob.glob(os.path.join(COCO_VAL_IMAGES, "*.png"))
        
        for image_path in image_paths:
            img = cv2.imread(image_path)
            outputs = self.predictor(img)
            instances = outputs["instances"].to("cpu")

            # confidence scores
            scores = instances.scores.numpy() 
            boxes = instances.pred_boxes.tensor.numpy()

            print(f'\n Boxes')
            print(len(boxes))

            for i, box in enumerate(boxes):
                if scores[i] > 0.5:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            output_path = os.path.join('output_predictions', os.path.basename(image_path))
            cv2.imwrite(output_path, img)



    def log_metrics(self, results, split_name="test"):
        """
        Log evaluation metrics to WandB.
        """
        metrics = {
            f"{split_name}_mAP": results["bbox"]["AP"], 
            f"{split_name}_AP50": results["bbox"]["AP50"],
            f"{split_name}_AP75": results["bbox"]["AP75"],
            f"{split_name}_AR": results["bbox"]["AR"],    
        }
        
        # Calculate F1 Score based on AP and AR
        precision = results["bbox"]["AP"] 
        recall = results["bbox"]["AR"]  
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        metrics[f"{split_name}_F1"] = f1_score
        
        wandb.log(metrics)

        # Plot F1 score
        plt.plot(f1_score, marker="o", label=f"{split_name}_F1 Score")
        plt.title(f"{split_name} F1 Score Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("F1 Score")
        plt.legend()
        wandb.log({f"{split_name}_F1_Score_Plot": wandb.Image(plt)})
        plt.clf() 

        for key, value in metrics.items():
            plt.plot(self.storage.iter, value, "go", label=key)
            plt.title(f"{key} Over Iterations")
            plt.xlabel("Epochs")
            plt.ylabel(key)
            plt.legend()
            plt.savefig(f"{split_name}_{key}_over_time.png")
            plt.clf()


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
        self.log_metrics(test_results, 'test')

        wandb.log(test_results)
        return test_results 