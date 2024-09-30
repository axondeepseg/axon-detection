from ultralytics import YOLO
from normalize_labels import normalize_labels

if __name__ == '__main__':
    # Load a model
    model = YOLO("./yolov8n.pt")

    # Normalize the labels
    normalize_labels()

    # Train the model
    model.train(data="../data.yaml", epochs=50, imgsz=640, device='mps')
