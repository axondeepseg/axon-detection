import glob
import os
import time
import cv2
import wandb as wandb
import matplotlib.pyplot as plt
import torch
import numpy as np
from detectron2.engine import DefaultTrainer, hooks
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
from pycocotools.cocoeval import COCOeval

try:
    from detectron2.evaluation.fast_eval_api import COCOeval_opt
except ImportError:
    COCOeval_opt = COCOeval


# from detectron2.evaluation.coco_evaluation import _evaluate_box_proposals

from detectron2.data.catalog import MetadataCatalog

# from detectron2.projects.DensePose.densepose.densepose_coco_evaluation import (
#     DensePoseCocoEval,
# )

from retinaNet.constants.data_file_constants import (
    COCO_TEST_REG_NAME,
    COCO_VAL_TEM_IMAGES,
    # COCO_VAL_SEM_IMAGES,
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

        # PREDICTION part for VAL PRECISION / RECALL

        try:
            final_val_metrics = self.evaluate()
            print(f"\nVAL METRICS ARE: {final_val_metrics}")
        except Exception as e:
            print("Validation run stopped due to:" + str(e))

        # PREDICTION part for VAL visualization of result

        # self.evaluate()

        current_lr = self.optimizer.param_groups[0]["lr"]
        print(
            f"\nLR at iteration={current_iteration} & epoch={current_iteration / 8}: {current_lr}"
        )

        # self.predictor.model.load_state_dict(self.model.state_dict())

        # image_paths = glob.glob(os.path.join(COCO_VAL_TEM_IMAGES, "*.png"))

        # for image_path in image_paths:
        #     print("image path")
        #     print(image_path)
        #     img = cv2.imread(image_path)
        #     outputs = self.predictor(img)
        #     instances = outputs["instances"].to("cpu")

        #     # confidence scores
        #     scores = instances.scores.numpy()
        #     boxes = instances.pred_boxes.tensor.numpy()

        #     print(f"\n Boxes")
        #     print(len(boxes))

        #     for i, box in enumerate(boxes):
        #         if scores[i] > CONF_THRESHOLD:
        #             x1, y1, x2, y2 = map(int, box)
        #             cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        #     output_path = os.path.join(
        #         "output_predictions", "modified_params_" + os.path.basename(image_path)
        #     )
        #     cv2.imwrite(output_path, img)
        #     wandb.log(
        #         {
        #             "Val Prediction": [
        #                 wandb.Image(img, caption=os.path.basename(image_path))
        #             ]
        #         }
        #     )
        #     break

    def log_metrics(self, results, split_name="test"):
        """
        Log evaluation metrics to WandB.
        """

        metrics = {
            f"{split_name}_AP_50:95": results["bbox"]["AP"],
            # f"{split_name}_AP50": results["bbox"]["AP50"],
            # f"{split_name}_AP75": results["bbox"]["AP75"],
        }

        wandb.log(metrics)

    def evaluate(self):
        # thing_classes = MetadataCatalog.get(COCO_VAL_ANNOTATION)
        # print('thing classes')
        # print(thing_classes)
        evaluator = COCOEvaluator(
            COCO_VAL_REG_NAME, output_dir="./output/", max_dets_per_image=1000
        )
        val_loader = build_detection_test_loader(self.cfg, COCO_VAL_REG_NAME)
        results = inference_on_dataset(self.model, val_loader, evaluator)

        print("\n VAL RESULTS")
        print(results)

        # These steps were to debug the printing of recall

        # try:
        #     evaluator.evaluate()
        # except Exception as e:
        #     print(f"Evaluate error: {e}")

        # try:
        #     evaluator.accumulate()
        # except Exception as e:
        #     print(f"Accumulate error: {e}")

        # try:
        #     evaluator.summarize()
        # except Exception as e:
        #     print(f"Summarize error: {e}")

        # print("=========================")
        # print(evaluator.stats)
        # print("=========================")

        # print("Results Val")
        # print(results)

        # print("coco eval results")
        # print(evaluator._results)

        # coco_eval = DensePoseCocoEval(coco_gt, coco_dt, "densepose", dpEvalMode=DensePoseEvalMode.GPSM)
        # coco_eval.evaluate()
        # coco_eval.accumulate()
        # coco_eval.summarize()

        self.log_metrics(results, "val")
        wandb.log(results)
        return results

    def visualize_predictions(
        self,
        test_dir,
        conf_threshold=CONF_THRESHOLD,
        output_directory="output_predictions",
    ):
        """
        Visualizes predictions on test images, saves the results, and logs them to WandB.

        Args:
            test_dir (str): Directory containing test images.
            conf_threshold (float): Confidence threshold for filtering predictions.
            output_directory (str): Directory to save prediction results.
        """
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Load the model's state dict into the predictor
        self.predictor.model.load_state_dict(self.model.state_dict())
        image_paths = glob.glob(os.path.join(test_dir, "*.png"))

        for image_path in image_paths:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to load image: {image_path}")
                continue

            print("Predicting on image:", image_path)

            start_time = time.time()
            outputs = self.predictor(img)
            inference_time = time.time() - start_time

            wandb.log({"Inference Time (s)": inference_time})
            print(f"Inference time for {image_path}: {inference_time:.4f} seconds")

            instances = outputs.get("instances")
            if instances is None:
                print(f"No predictions found for image: {image_path}")
                continue

            instances = instances.to("cpu")
            boxes = (
                instances.pred_boxes.tensor.numpy()
                if instances.has("pred_boxes")
                else []
            )
            scores = instances.scores.numpy() if instances.has("scores") else []

            print(f"Number of boxes detected: {len(boxes)}")

            for i, box in enumerate(boxes):
                if scores[i] > conf_threshold:
                    x1, y1, x2, y2 = map(int, box)

                    green = int(255 * (1 - scores[i]))
                    blue = int(255 * scores[i])
                    color = (0, green, blue)

                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)

                    label = f"{scores[i]:.2f}"
                    font_scale = 0.5
                    font_thickness = 1
                    text_size = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                    )[0]
                    text_x = x1
                    text_y = y1 - 5
                    text_y = max(text_y, 10)

                    cv2.rectangle(
                        img,
                        (text_x, text_y - text_size[1]),
                        (text_x + text_size[0], text_y),
                        color,
                        -1,
                    )
                    cv2.putText(
                        img,
                        label,
                        (text_x, text_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (255, 255, 255),
                        font_thickness,
                    )

            output_path = os.path.join(output_directory, os.path.basename(image_path))
            success = cv2.imwrite(output_path, img)
            if success:
                print(f"Saved prediction image to {output_path}")
            else:
                print(f"Failed to save prediction image to {output_path}")

            wandb.log(
                {
                    f"Test Prediction for {image_path}": [
                        wandb.Image(img, caption=os.path.basename(image_path))
                    ]
                }
            )
            break

    def test(self):
        test_evaluator = COCOEvaluator(
            COCO_TEST_REG_NAME, output_dir="./output/", max_dets_per_image=1000
        )
        test_loader = build_detection_test_loader(self.cfg, COCO_TEST_REG_NAME)

        test_results = inference_on_dataset(
            self.model,
            test_loader,
            test_evaluator,
        )
        self.log_metrics(test_results, "test")

        print("\ntest results")
        print(test_results)

        wandb.log(test_results)
        return test_results
