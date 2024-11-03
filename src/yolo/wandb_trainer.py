import time
import wandb
from ultralytics import YOLO

class WandbTrainer:
    def __init__(self, model_path, config):
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
        wandb.log({"training_time": training_time})
