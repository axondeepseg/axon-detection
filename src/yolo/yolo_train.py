from ultralytics import YOLO
# import wandb

WANDB_PROJECT = "mnist-viz"

if __name__ == '__main__':

    # start a new wandb run to track this script
    # wandb.init(
    #     # set wandb project where this run will be logged
    #     project="my-awesome-project",

    #     # track hyperparameters and run metadata
    #     config={
    #         "epochs": 3,
    #     }
    # )
    
    # Load a model
    model = YOLO("./yolov8n.pt")

    # Train the model
    model.train(data="data.yaml", epochs=3, imgsz=640, device='mps')
