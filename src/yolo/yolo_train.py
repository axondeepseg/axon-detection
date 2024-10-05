from ultralytics import YOLO
from normalize_labels import normalize_labels

if __name__ == '__main__':
    model = YOLO("./yolov8n.pt")
    # Best results observed with 138 epochs
    model.train(data="../data.yaml", epochs=138, imgsz=640, device='mps')
