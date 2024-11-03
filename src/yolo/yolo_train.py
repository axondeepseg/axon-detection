import wandb
import os
from ..constants.wandb_yolo_constants import WANDB_ENTITY, WANDB_PROJECT, WANDB_RUN_NAME, WANDB_RUN_ID
from ..yolo.wandb_trainer import WandbTrainer
  

if __name__ == "__main__":

    config = {
        # NOTE: "datasets_dir" in settings.json of Ultralytics should look like this:  "\\axon-detection"
        'data': 'src/data-yolo/data.yaml',
        'epochs': 138,
        'imgsz': 640,
        'batch': 16,
        'project': WANDB_PROJECT,
        'name': WANDB_RUN_NAME
    }

    trainer = WandbTrainer(model_path="./yolov8n.pt", config=config)
    trainer.run_step()

    # Log inference time
    trainer.log_inference_time(test_dir='src/data-yolo/images/test')  

    # Visualize predictions
    trainer.visualize_predictions(test_dir='src/data-yolo/images/test')

    wandb.finish()
