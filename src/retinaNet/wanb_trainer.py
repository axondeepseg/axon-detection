import glob
import os
import time
import cv2
import wandb as wandb
import matplotlib as plt
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
        
        # Get last registered metrics (loss, learning rate, etc)
        metrics_dict = {key: float(value[0]) for key, value in self.storage._latest_scalars.items()}

        for key, value in metrics_dict.items():
            wandb.log({key: value})
        
        current_lr = self.optimizer.param_groups[0]["lr"]
        wandb.log({"learning_rate": current_lr})
        
        training_time = time.time() - self.start_time
        wandb.log({"training_time": training_time})


        # PREDICTION part for devugging
        self.predictor.model.load_state_dict(self.model.state_dict())

        print('\nstate model')
        print(self.model.state_dict)
        image_paths = glob.glob(os.path.join(COCO_VAL_IMAGES, "*.png"))
        
        for image_path in image_paths:
            img = cv2.imread(image_path)
            outputs = self.predictor(img)
            instances = outputs["instances"].to("cpu")

            boxes = instances.pred_boxes.tensor.numpy()
            print(f'\n Boxes')
            print(len(boxes))
            print(boxes)

            filtered_boxes = [box for i, box in enumerate(boxes)]

            for i, box in enumerate(filtered_boxes):
                print('filtered boxes sizes')
                print(len(filtered_boxes))

                x1, y1, x2, y2 = map(int, box)
                # confidence = box.conf[0]
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # cv2.putText(img, f"{confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            output_path = os.path.join('output_predictions', os.path.basename(image_path))
            success = cv2.imwrite(output_path, img)
            if success:
                print(f"Saved prediction image to {output_path}")
            else:
                print(f"Failed to save prediction image to {output_path}")


    def log_metrics(self, results, split_name="Test"):
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