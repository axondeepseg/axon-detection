import json
import os
import cv2
import matplotlib.pyplot as plt
from retinaNet.constants.data_file_constants import OUTPUT_TRUE_LABELS

def visualize_true_labels(annotations_path, set_type='test'):

    output_directory = f"{OUTPUT_TRUE_LABELS}/{set_type}"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    with open(annotations_path, 'r') as f:
        data = json.load(f)

    # Load the image file path from the JSON data
    images_info = data['images']

    for image_info in images_info:
        image_path = image_info['file_name']
        image = cv2.imread(f"data-coco/images/{set_type}/{image_path}")

        # Draw bounding boxes
        for annotation in data['annotations']:
            bbox = annotation['bbox']
            category_id = annotation['category_id']
            
            # Bounding box coordinates
            x, y, width, height = bbox
            x2, y2 = int(x + width), int(y + height)
            
            # Draw rectangle and label
            cv2.rectangle(image, (int(x), int(y)), (x2, y2), (255, 0, 0), 2)

        output_path = os.path.join(output_directory, os.path.basename(image_path))
        success = cv2.imwrite(output_path, image)