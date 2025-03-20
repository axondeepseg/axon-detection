import json
import os
import cv2
import matplotlib.pyplot as plt
from retinaNet.constants.data_file_constants import OUTPUT_TRUE_LABELS


def visualize_true_labels(annotations_path, data_type="sem", set_type="test"):

    output_directory = f"{OUTPUT_TRUE_LABELS}/{data_type}/{set_type}"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    print("annotations")
    print(annotations_path)

    with open(annotations_path, "r") as f:
        data = json.load(f)

    # Load the image file path from the JSON data
    images_info = data["images"]

    print("images info")
    print(data["images"])

    for image_info in images_info:
        print(f"\nImage info: {image_info}")
        image_id = image_info["id"]
        image_path = image_info["file_name"]
        print("output path")
        print(f"data-coco/{data_type}/images/{set_type}/{image_path}")
        image = cv2.imread(f"data-coco/{data_type}/images/{set_type}/{image_path}")

        annotations = filter(
            lambda annotation: annotation["image_id"] == image_id, data["annotations"]
        )

        for annotation in annotations:
            bbox = annotation["bbox"]
            x, y, width, height = bbox
            x2, y2 = int(x + width), int(y + height)
            cv2.rectangle(image, (int(x), int(y)), (x2, y2), (0, 255, 0), thickness=7)

        output_path = os.path.join(output_directory, os.path.basename(image_path))

        print("output_path")
        print(output_path)
        try:
            success = cv2.imwrite(output_path, image)
        except Exception as e:
            print(f"error {e}")
