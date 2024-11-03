import time
import wandb
from ultralytics import YOLO
import glob
import os
import cv2
from ..constants.wandb_yolo_constants import WANDB_ENTITY, WANDB_PROJECT, WANDB_RUN_NAME, WANDB_RUN_ID

class WandbTrainer:
    def __init__(self, model_path, config):
        self.model = YOLO(model_path)
        self.cfg = config  
        self.start_time = time.time()
        wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, name=WANDB_RUN_NAME, id=WANDB_RUN_ID)

    def __del__(self):
        if wandb.run:
            wandb.finish()
    
    def run_step(self):
        results = self.model.train(
            data=self.cfg['data'],
            epochs=self.cfg['epochs'],
            imgsz=self.cfg['imgsz'],
            batch=self.cfg['batch'],
            project=self.cfg['project'],
            name=self.cfg['name'],
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

        # self.log_inference_time(test_dir="src/data-yolo/images/test") # log inference time on test set
        self.visualize_predictions(test_dir="src/data-yolo/images/test", conf=0.25) # visualize predictions on test set


    def log_inference_time(self, test_dir):
        image_paths = glob.glob(os.path.join(test_dir, "*.png")) 
        start_inference = time.time()
        
        for image_path in image_paths:
            _ = self.model.predict(image_path)
        
        inference_time = time.time() - start_inference
        wandb.init(mode="disabled")
        wandb.log({"inference_time": inference_time})
        print(f"Inference time on test set: {inference_time:.2f} seconds")

    def visualize_predictions(self, test_dir, conf=0.25):
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
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(img, f"{confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                output_path = os.path.join(output_directory, os.path.basename(image_path))
                success = cv2.imwrite(output_path, img)
                if success:
                    print(f"Saved prediction image to {output_path}")
                else:
                    print(f"Failed to save prediction image to {output_path}")
                
                wandb.log({"Prediction": [wandb.Image(img, caption=os.path.basename(image_path))]})
        
        print("Predictions visualized and logged to wandb.")

