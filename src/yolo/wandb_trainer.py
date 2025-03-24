import time
import wandb
from ultralytics import YOLO
import glob
import os
import cv2
from constants.wandb_yolo_constants import WANDB_ENTITY, WANDB_PROJECT, WANDB_RUN_NAME, WANDB_RUN_ID

IMAGES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data-yolo', 'images', 'test')

class WandbTrainer:
    def __init__(self, model_path, config):
        self.model = YOLO(model_path)
        self.cfg = config  
        self.start_time = time.time()
        wandb.login()
        wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, name=WANDB_RUN_NAME, mode="online")

    # def __del__(self):
    #     if wandb.run:
    #         wandb.finish()
    
    def run_step(self):
        results = self.model.train(
            data=self.cfg['data'],
            epochs=self.cfg['epochs'],
            imgsz=self.cfg['imgsz'],
            batch=self.cfg['batch'],
            project=self.cfg['project'],
            name=self.cfg['name'],
            patience=1000, # to prevent early stopping if necessary
            exist_ok=True
        )
        
        metrics = results.results_dict
        
        metrics_dict = {
            "train/loss": metrics.get('train/loss', None),
            "val/loss": metrics.get('val/loss', None),
            "precision": metrics.get('precision', None),
            "recall": metrics.get('recall', None),
            "mAP_0.5": metrics.get('mAP_0.5', None),
            "mAP_0.5:0.95": metrics.get('mAP_0.5:0.95', None)
        }
        
        for key, value in metrics_dict.items():
            if value is not None:
                wandb.init(mode="disabled") 
                wandb.log({key: value})
        
        training_time = time.time() - self.start_time
        wandb.init(mode="disabled")
        wandb.log({"training_time": training_time})

        # log inference time on test set
        # self.log_inference_time(test_dir=IMAGES_PATH)

        # visualize predictions on test set
        self.visualize_predictions(test_dir=IMAGES_PATH) 


    def log_inference_time(self, test_dir):
        image_paths = glob.glob(os.path.join(test_dir, "*.png")) 
        start_inference = time.time()
        
        for image_path in image_paths:
            _ = self.model.predict(image_path)
        
        inference_time = time.time() - start_inference
        wandb.init(mode="disabled")
        wandb.log({"inference_time": inference_time})
        print(f"Inference time on test set: {inference_time:.2f} seconds")

    def visualize_predictions(self, test_dir, conf=0.6):
        output_directory = 'output_predictions'
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        image_paths = glob.glob(os.path.join(test_dir, "*.png"))

        for image_path in image_paths:
            print("Predicting on image:", image_path)
            
            results = self.model.predict(image_path, save=False, conf=conf)

            for result in results:
                print("Visualizing prediction...")
                img = result.orig_img

                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=9)
                    cv2.putText(img, f"{confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                output_path = os.path.join(output_directory, os.path.basename(image_path))
                success = cv2.imwrite(output_path, img)
                if success:
                    print(f"Saved prediction image to {output_path}")
                else:
                    print(f"Failed to save prediction image to {output_path}")

                wandb.log({"Prediction": [wandb.Image(img, caption=os.path.basename(image_path))]})

        print("Predictions visualized and logged to wandb.")


    def visualize_ground_truth(self, test_dir, labels_dir):
        output_directory = 'output_ground_truth'
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        image_paths = glob.glob(os.path.join(test_dir, "*.png"))

        for image_path in image_paths:
            print("Visualizing ground truth for image:", image_path)
            
            # reading the corresponding label file
            label_file = os.path.join(labels_dir, os.path.splitext(os.path.basename(image_path))[0] + ".txt")
            if not os.path.exists(label_file):
                print(f"No label file found for {image_path}, skipping...")
                continue
            
            # loading image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to load image {image_path}, skipping...")
                continue

            height, width, _ = img.shape

            # reading label file and draw bounding boxes
            with open(label_file, "r") as f:
                for line in f:
                    class_id, x_center, y_center, box_width, box_height = map(float, line.strip().split())
                    
                    # converting YOLO format to pixel coordinates
                    x1 = int((x_center - box_width / 2) * width)
                    y1 = int((y_center - box_height / 2) * height)
                    x2 = int((x_center + box_width / 2) * width)
                    y2 = int((y_center + box_height / 2) * height)

                    # drawing rectangle and class label
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # saving the image with ground truth visualization
            output_path = os.path.join(output_directory, os.path.basename(image_path))
            success = cv2.imwrite(output_path, img)
            if success:
                print(f"Saved ground truth image to {output_path}")
            else:
                print(f"Failed to save ground truth image to {output_path}")

            # logging to wandb
            wandb.log({"Ground Truth": [wandb.Image(img, caption=os.path.basename(image_path))]})

        print("Ground truth visualized and logged to wandb.")
        
        
    def evaluate_model(self):
        wandb.log({"test_value": "hello"})
        print("Evaluating model on test set...")

        results = self.model.val(data=self.cfg['data'], split="test")

        # Extract evaluation metrics
        metrics_dict = results.results_dict
        print(f"results.results_dict: {results.results_dict}")
        
        ap_50 = metrics_dict.get('metrics/mAP50(B)', None)
        ap_50_95 = metrics_dict.get('metrics/mAP50-95(B)', None)
        ar = metrics_dict.get('metrics/recall(B)', None)

        print(f"AP @ 0.5: {ap_50}")
        print(f"AP @ 0.5:0.95: {ap_50_95}")
        print(f"AR: {ar}")
        print("WandB Run :", wandb.run)
        print("WandB Run URL:", wandb.run.get_url())


        # Log to Weights & Biases
        wandb.log({
            "AP@0.5": ap_50,
            "AP@0.5:0.95": ap_50_95,
            "AR": ar
        })
        
        print("WandB Run Details:")
        print(f"Run ID: {wandb.run.id}")
        print(f"Run Name: {wandb.run.name}")
        print(f"Run URL: {wandb.run.get_url()}")
        print(f"Project: {wandb.run.project}")

        print("Evaluation complete and metrics logged to wandb.")
        
        wandb.finish()

