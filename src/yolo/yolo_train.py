from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("./yolov8n.pt")
    # Best results observed with 138 epochs
    model.train(data="../data.yaml", epochs=138, imgsz=640, device='mps')
