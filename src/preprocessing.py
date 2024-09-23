import random
import numpy as np
import subprocess
import utils
import cv2
import os
import shutil
import json
from tqdm import tqdm

from utils import data_split

SEM_DATASET_URL = "https://github.com/axondeepseg/data_axondeepseg_sem"

# Reorganizes data structure into COCO (retinanet)

# Other problem: Bounding box surrounding myelin and axon (only axon)


def save_yolo_dset(image_mask_pairs, image_dir, mask_dir):
    for image_name, img, mask_path in image_mask_pairs:
        # Save the image
        cv2.imwrite(os.path.join(image_dir, image_name), img)
        
        # Copy the corresponding label (mask) file
        shutil.copy(mask_path, os.path.join(mask_dir, os.path.basename(mask_path)))


def save_coco_dset(image_mask_pairs, image_dir, annotations):
    for image_name, img, image_info, axon_annotations, myelin_annotations in image_mask_pairs:
        # Save image
        cv2.imwrite(os.path.join(image_dir, image_name), img)
        # Add to annotations
        annotations["images"].append(image_info)
        annotations["annotations"].extend(axon_annotations)
        annotations["annotations"].extend(myelin_annotations)

def download_default_sem_dataset():
    if not os.path.exists("data_axondeepseg_sem"):
        subprocess.run(["git", "clone", SEM_DATASET_URL])

def preprocess_data_yolo(data_dir: str = "data_axondeepseg_sem"):
    """Preprocesses the loaded BIDS data for object detection.

    Steps:
    1. Load
    2. Normalize + windowing
    3. find regions axons Use skimage.measures.regionprops()
    4. Croping + Resizing

    LABELS FORMAT DESCRIPTION FOR YOLO: similar to COCO
    - One *.txt file per image
    - One row per object in class x_center y_center width height format.
    - Box coordinates must be in normalized xywh format (from 0 to 1)
    - Class numbers should be zero-indexed
    """

    # TODO: Issue of raw path
    processed_images_dir = "data-yolo/images"
    processed_masks_dir = "data-yolo/labels"

    train_images_dir = os.path.join(processed_images_dir, "train")
    train_masks_dir = os.path.join(processed_masks_dir, "train")
    val_images_dir = os.path.join(processed_images_dir, "val")
    val_masks_dir = os.path.join(processed_masks_dir, "val")
    test_images_dir = os.path.join(processed_images_dir, "test")
    test_masks_dir = os.path.join(processed_masks_dir, "test")

    if data_dir == "data_axondeepseg_sem":
        download_default_sem_dataset()

    #Creating directories
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_masks_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_masks_dir, exist_ok=True)
    os.makedirs(test_images_dir, exist_ok=True)
    os.makedirs(test_masks_dir, exist_ok=True)

    data_dict = utils.load_bids_images(data_dir)
    bbox_data = []

    #Collect images and masks in a format for YOLO
    image_mask_pairs = []

    for subject in tqdm(data_dict.keys(), desc='Loading dataset for YOLO conversion.'):
        if subject == "sidecar":
            continue

        pixel_size = data_dict[subject]['sidecar']
        for sample in data_dict[subject].keys():
            if sample == "sidecar":
                continue
            img_path = data_dict[subject][sample]['image']
            axon_seg_path = data_dict[subject][sample]['axon']
            myelin_seg_path = data_dict[subject][sample]['myelin']
            image_name = f"{subject}_{sample}.png"
            label_name = f"{subject}_{sample}.txt"

            # Load and preprocess the images and the segmentation mask
            img = utils.load_bids_image(img_path, pixel_size)
            img = utils.normalize_and_window(img)

            # Load segmentation masks and find regions
            axon_seg = cv2.imread(axon_seg_path, cv2.IMREAD_GRAYSCALE)
            axon_seg_regions = utils.find_regions(axon_seg)
            for i, region in enumerate(axon_seg_regions):
                minr, minc, maxr, maxc = region.bbox
                bbox_data.append({"image_name": f"{subject}_{sample}.png", "xmin": minc, "ymin": minr, "xmax": maxc, "ymax": maxr, "class": "axon"})

            # Process Myelin
            myelin_seg = cv2.imread(myelin_seg_path, cv2.IMREAD_GRAYSCALE)
            myelin_seg_regions = utils.find_regions(myelin_seg)
            with open(os.path.join(processed_masks_dir, label_name), "w") as file:
                for i, region in enumerate(myelin_seg_regions):
                    minr, minc, maxr, maxc = region.bbox
                    centroidx, centroidy = region.centroid[0], region.centroid[1]
                    width,height = region.axis_major_length, region.axis_minor_length
                    bbox_data.append({"image_name": f"{subject}_{sample}.png", "xmin": minc, "ymin": minr, "xmax": maxr, "ymax": maxc, "class": "myelin"})

                    # Normalize coordinates
                    img_height, img_width = img.shape[:2]
                    centroidx /= img_width
                    centroidy /= img_height
                    width /= img_width
                    height /= img_height
                    file.write('0 {} {} {} {}\n'.format(centroidx, centroidy, width, height))

            # Add image and Masks paths to the list for split
            image_mask_pairs.append((image_name, img, os.path.join(processed_masks_dir, label_name)))

    data_split = data_split(image_mask_pairs)
    
    yolo_data = []
    for (image_name, img, mask_path) in image_mask_pairs:
        if image_name in data_split['train'] + data_split['val'] + data_split['test']:
            yolo_data.append((image_name, img, mask_path))

    yolo_train_set = [entry for entry in yolo_data if entry[0] in data_split['train']]
    yolo_val_set = [entry for entry in yolo_data if entry[0] in data_split['val']]
    yolo_test_set = [entry for entry in yolo_data if entry[0] in data_split['test']]

    save_yolo_dset(yolo_train_set, train_images_dir, train_masks_dir)
    save_yolo_dset(yolo_val_set, val_images_dir, val_masks_dir)
    save_yolo_dset(yolo_test_set, test_images_dir, test_masks_dir)

def preprocess_data_coco(data_dir: str = "data_axondeepseg_sem"):
    """Preprocesses the loaded BIDS data for object detection and converts it into COCO format.

    Steps:
    1. Load
    2. Normalize + windowing
    3. Find regions (axons and myelin) using skimage.measure.regionprops()
    4. Create COCO annotations
    5. Croping + Resizing
    6. Save the annotations and images in the appropriate COCO directories.
    """

    processed_images_dir = "data-coco/images"
    processed_annotations_dir = "data-coco/annotations"

    train_images_dir = os.path.join(processed_images_dir, "train")
    val_images_dir = os.path.join(processed_images_dir, "val")
    test_images_dir = os.path.join(processed_images_dir, "test")

    train_annotations_file = os.path.join(processed_annotations_dir, "json_annotation_train.json")
    val_annotations_file = os.path.join(processed_annotations_dir, "json_annotation_val.json")
    test_annotations_file = os.path.join(processed_annotations_dir, "json_annotation_test.json")

    if data_dir == "data_axondeepseg_sem":
        download_default_sem_dataset()

    # Create directories
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(test_images_dir, exist_ok=True)
    os.makedirs(processed_annotations_dir, exist_ok=True)

    data_dict = utils.load_bids_images(data_dir)

    annotation_id = 1
    image_id = 1
    image_mask_pairs = []

    # Structures for COCO annotations
    common_annotations = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "axon", "supercategory": "cell"},
            {"id": 2, "name": "myelin", "supercategory": "cell"},
        ],
    }

    train_annotations = common_annotations.copy()
    val_annotations = common_annotations.copy()
    test_annotations = common_annotations.copy()

    for subject in tqdm(data_dict.keys(), desc='Loading dataset for COCO conversion.'):
        if subject == "sidecar":
            continue

        pixel_size = data_dict[subject]['sidecar']
        for sample in data_dict[subject].keys():
            if sample == "sidecar":
                continue
            img_path = data_dict[subject][sample]['image']
            axon_seg_path = data_dict[subject][sample]['axon']
            myelin_seg_path = data_dict[subject][sample]['myelin']
            image_name = f"{subject}_{sample}.png"

            # Load and preprocess the images
            img = utils.load_bids_image(img_path, pixel_size)
            img = utils.normalize_and_window(img)

            # Save image metadata for COCO
            img_height, img_width = img.shape[:2]
            image_info = {
                "id": image_id,
                "width": img_width,
                "height": img_height,
                "file_name": image_name,
            }

            # Load segmentation masks and find regions
            axon_seg = cv2.imread(axon_seg_path, cv2.IMREAD_GRAYSCALE)
            axon_seg_regions = utils.find_regions(axon_seg)
            axon_annotations = []
            for region in axon_seg_regions:
                minr, minc, maxr, maxc = region.bbox
                bbox_width = maxc - minc
                bbox_height = maxr - minr
                bbox_area = bbox_width * bbox_height
                axon_annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,  # Axon category
                    "bbox": [minc, minr, bbox_width, bbox_height],
                    "area": bbox_area,
                    "iscrowd": 0,
                })
                annotation_id += 1

            # Process Myelin
            myelin_seg = cv2.imread(myelin_seg_path, cv2.IMREAD_GRAYSCALE)
            myelin_seg_regions = utils.find_regions(myelin_seg)
            myelin_annotations = []
            for region in myelin_seg_regions:
                minr, minc, maxr, maxc = region.bbox
                bbox_width = maxc - minc
                bbox_height = maxr - minr
                bbox_area = bbox_width * bbox_height
                myelin_annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 2,  # Myelin category
                    "bbox": [minc, minr, bbox_width, bbox_height],
                    "area": bbox_area,
                    "iscrowd": 0,
                })
                annotation_id += 1

            image_mask_pairs.append((image_name, img, image_info, axon_annotations, myelin_annotations))

            # Increment image_id for next image
            image_id += 1

    data_split = data_split(image_mask_pairs)
    
    coco_data = []
    for (image_name, img, image_info, axon_annotations, myelin_annotations) in image_mask_pairs:
        if image_name in data_split['train'] + data_split['val'] + data_split['test']:
            coco_data.append((image_name, img, image_info, axon_annotations, myelin_annotations))

    coco_train_set = [entry for entry in coco_data if entry[0] in data_split['train']]
    coco_val_set = [entry for entry in coco_data if entry[0] in data_split['val']]
    coco_test_set = [entry for entry in coco_data if entry[0] in data_split['test']]

    save_coco_dset(coco_train_set, train_images_dir, train_annotations)
    save_coco_dset(coco_val_set, val_images_dir, val_annotations)
    save_coco_dset(coco_test_set, test_images_dir, test_annotations)

    # Save COCO annotations to respective files
    with open(train_annotations_file, "w") as f:
        json.dump(train_annotations, f)
    with open(val_annotations_file, "w") as f:
        json.dump(val_annotations, f)
    with open(test_annotations_file, "w") as f:
        json.dump(test_annotations, f)


if __name__ == '__main__':
    split_file = 'data_sem_split.json'
    if os.path.exists(split_file):
        os.remove(split_file)
        print(f"{split_file} has been deleted.")
    else:
        print(f"{split_file} does not exist.")
        
    preprocess_data_yolo()
    preprocess_data_coco()
