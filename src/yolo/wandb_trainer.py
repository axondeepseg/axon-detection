import time
import wandb
from ultralytics import YOLO
import glob
import os
from ..constants.wandb_yolo_constants import WANDB_ENTITY, WANDB_PROJECT, WANDB_RUN_NAME, WANDB_RUN_ID

class WandbTrainer:
    def __init__(self, model_path, config):
        wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, name=WANDB_RUN_NAME, id=WANDB_RUN_ID)
        self.model = YOLO(model_path)
        self.cfg = config  
        self.start_time = time.time()
    
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
                wandb.log({key: value})
        
        training_time = time.time() - self.start_time
        wandb.init(mode="disabled")
        wandb.log({"training_time": training_time})


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
        image_paths = glob.glob(os.path.join(test_dir, "*.png")) 
        
        for image_path in image_paths:
            results = self.model.predict(image_path, save=False, conf=conf)
            
            for result in results:
                result_plotted = result.plot()
                wandb.init(mode="disabled") 
                wandb.log({"Prediction": [wandb.Image(result_plotted, caption=os.path.basename(image_path))]})
        
        print("Predictions visualized and logged to wandb.")
