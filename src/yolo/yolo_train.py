from ..constants.wandb_yolo_constants import WANDB_ENTITY, WANDB_PROJECT, WANDB_RUN_NAME, WANDB_RUN_ID
from ..yolo.wandb_trainer import WandbTrainer
  

if __name__ == "__main__":

    config = {
        # NOTE: "datasets_dir" in settings.json of Ultralytics should look like this:  "\\axon-detection"
        'data': 'src/data-yolo/data.yaml',
        'epochs': 250,
        'imgsz': 640,
        'optimizer': 'adam',
        'rect': True,
        'batch': 16,
        'project': WANDB_PROJECT,
        'name': WANDB_RUN_NAME
    }

    trainer = WandbTrainer(model_path="./yolov8n.pt", config=config)    
    trainer.run_step()
    # example usage to visualize ground truth
    # trainer.visualize_ground_truth(test_dir="src/data-yolo/images/val", labels_dir="src/data-yolo/labels/val")
    

