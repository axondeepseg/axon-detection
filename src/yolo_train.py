from ultralytics import YOLO


if __name__ == '__main__':
    # Load a model
    model = YOLO("./yolov8n.yaml")

    # Train the model
    model.train(data="data.yaml", epochs=10, imgsz=640, device='mps')
