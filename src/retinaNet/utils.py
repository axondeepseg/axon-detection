import json
import csv

def get_bbox_csv(annotations_path, set_type='val'):

    with open(annotations_path, "r") as json_file:
        json_data = json.load(json_file)

    image_map = {img["id"]: img["file_name"] for img in json_data["images"]}

    csv_file_path = f"retinaNet/data_csv/annotations_{set_type}.csv"

    with open(csv_file_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["path/to/image.jpg", "x1", "y1", "x2", "y2", "class_name"])
        
        for annotation in json_data["annotations"]:
            image_id = annotation["image_id"]
            file_name = image_map.get(image_id, "")
            bbox = annotation["bbox"]
            
            x1, y1, width, height = bbox
            
            # FIXME: Hardcoded class value of 0
            writer.writerow([file_name, x1, y1, width, height, 0])

if __name__ == '__main__':
    get_bbox_csv('data-coco/annotations/json_annotation_val.json')
