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
from detectron2.engine import DefaultPredictor

# from detectron2.evaluation.coco_evaluation import _evaluate_box_proposals

from detectron2.data.catalog import MetadataCatalog
from retinaNet.constants.data_file_constants import (
    COCO_TEST_REG_NAME,
    # COCO_VAL_TEM_IMAGES,
    COCO_VAL_SEM_IMAGES,
    COCO_VAL_REG_NAME,
)
from retinaNet.constants.config_constants import CONF_THRESHOLD


class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.start_time = time.time()
        self.predictor = DefaultPredictor(self.cfg)

    def run_step(self):
        super().run_step()

        metrics_dict = {
            key: float(value[0]) for key, value in self.storage._latest_scalars.items()
        }

        # Plot all registered metrics (loss, learning rate, etc)
        current_iteration = self.storage.iter
        for key, value in metrics_dict.items():
            wandb.log({key: value})

        current_lr = self.optimizer.param_groups[0]["lr"]
        wandb.log({"learning_rate": current_lr})

        training_time = time.time() - self.start_time
        wandb.log({"training_time": training_time})

        # PREDICTION part for visualization of result

        # self.evaluate()

        current_lr = self.optimizer.param_groups[0]["lr"]
        print(
            f"\nLR at iteration={current_iteration} & epoch={current_iteration / 8}: {current_lr}"
        )

        self.predictor.model.load_state_dict(self.model.state_dict())

        image_paths = glob.glob(os.path.join(COCO_VAL_SEM_IMAGES, "*.png"))

        for image_path in image_paths:
            print("image path")
            print(image_path)
            img = cv2.imread(image_path)
            outputs = self.predictor(img)
            instances = outputs["instances"].to("cpu")

            # confidence scores
            scores = instances.scores.numpy()
            boxes = instances.pred_boxes.tensor.numpy()

            print(f"\n Boxes")
            print(len(boxes))

            print("scores len")
            print(len(scores))

            for i, box in enumerate(boxes):
                if scores[i] > CONF_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            output_path = os.path.join(
                "output_predictions", "modified_params_" + os.path.basename(image_path)
            )
            cv2.imwrite(output_path, img)
            break

    def log_metrics(self, results, split_name="test"):
        """
        Log evaluation metrics to WandB.
        """

        metrics = {
            f"{split_name}_mAP": results["bbox"]["AP"],
            f"{split_name}_AP50": results["bbox"]["AP50"],
            f"{split_name}_AP75": results["bbox"]["AP75"],
        }

        wandb.log(metrics)

    def evaluate(self):
        # thing_classes = MetadataCatalog.get(COCO_VAL_ANNOTATION)
        # print('thing classes')
        # print(thing_classes)
        evaluator = COCOEvaluator(
            COCO_VAL_REG_NAME, output_dir="./output/", max_dets_per_image=2000
        )
        val_loader = build_detection_test_loader(self.cfg, COCO_VAL_REG_NAME)
        results = inference_on_dataset(self.model, val_loader, evaluator)

        self.log_metrics(results, "val")
        wandb.log(results)
        return results

    def test(self):
        test_evaluator = COCOEvaluator(
            COCO_TEST_REG_NAME, output_dir="./output/", max_dets_per_image=2000
        )
        test_loader = build_detection_test_loader(self.cfg, COCO_TEST_REG_NAME)

        test_results = inference_on_dataset(
            self.model,
            test_loader,
            test_evaluator,
        )
        self.log_metrics(test_results, "test")

        wandb.log(test_results)
        return test_results
