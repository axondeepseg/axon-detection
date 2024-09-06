import random
import subprocess
import utils
import cv2
import os
import shutil
import json

def preprocess_data_yolo():
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
    data_repo_url = "https://github.com/axondeepseg/data_axondeepseg_sem"
    data_dir = "data_axondeepseg_sem"  # Local directory for the cloned repository
    processed_images_dir = "dataset/images"
    processed_masks_dir = "dataset/labels"

    train_images_dir = os.path.join(processed_images_dir, "train")
    train_masks_dir = os.path.join(processed_masks_dir, "train")
    val_images_dir = os.path.join(processed_images_dir, "val")
    val_masks_dir = os.path.join(processed_masks_dir, "val")
    test_images_dir = os.path.join(processed_images_dir, "test")
    test_masks_dir = os.path.join(processed_masks_dir, "test")

    if not os.path.exists(data_dir):
        subprocess.run(["git", "clone", data_repo_url])

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

    for subject in data_dict.keys():
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

    #Shuffle and split the dataset
    random.shuffle(image_mask_pairs)
    num_train = 7
    num_val = 2
    num_test = 1

    train_set = image_mask_pairs[:num_train]
    val_set = image_mask_pairs[num_train:num_train + num_val]
    test_set = image_mask_pairs[num_train + num_val:num_train + num_val + num_test]

    def save_set(image_mask_pairs, image_dir, mask_dir):
        for image_name, img, mask_path in image_mask_pairs:
            cv2.imwrite(os.path.join(image_dir, image_name), img)
            shutil.copy(mask_path, os.path.join(mask_dir, os.path.basename(mask_path)))

    save_set(train_set, train_images_dir, train_masks_dir)
    save_set(val_set, val_images_dir, val_masks_dir)
    save_set(test_set, test_images_dir, test_masks_dir)


def preprocess_data_retinanet():
    """Preprocesses the loaded BIDS data for object detection and converts it into COCO format.

    Steps:
    1. Load
    2. Normalize + windowing
    3. Find regions (axons and myelin) using skimage.measure.regionprops()
    4. Create COCO annotations
    5. Croping + Resizing
    6. Save the annotations and images in the appropriate COCO directories.
    """

    data_repo_url = "https://github.com/axondeepseg/data_axondeepseg_sem"
    data_dir = "data_axondeepseg_sem"  # Local directory for the cloned repository
    processed_images_dir = "coco/images"
    processed_annotations_dir = "coco/annotations"

    train_images_dir = os.path.join(processed_images_dir, "train")
    val_images_dir = os.path.join(processed_images_dir, "val")
    test_images_dir = os.path.join(processed_images_dir, "test")

    train_annotations_file = os.path.join(processed_annotations_dir, "instances_train.json")
    val_annotations_file = os.path.join(processed_annotations_dir, "instances_val.json")
    test_annotations_file = os.path.join(processed_annotations_dir, "instances_test.json")

    if not os.path.exists(data_dir):
        subprocess.run(["git", "clone", data_repo_url])

    # Create directories
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(test_images_dir, exist_ok=True)
    os.makedirs(processed_annotations_dir, exist_ok=True)

    data_dict = utils.load_bids_images(data_dir)
    coco_annotations = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "axon", "supercategory": "cell"},
            {"id": 2, "name": "myelin", "supercategory": "cell"},
        ],
    }

    annotation_id = 1
    image_id = 1
    image_mask_pairs = []

    for subject in data_dict.keys():
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
            coco_annotations["images"].append({
                "id": image_id,
                "width": img_width,
                "height": img_height,
                "file_name": image_name,
            })

            # Load segmentation masks and find regions
            axon_seg = cv2.imread(axon_seg_path, cv2.IMREAD_GRAYSCALE)
            axon_seg_regions = utils.find_regions(axon_seg)
            for region in axon_seg_regions:
                minr, minc, maxr, maxc = region.bbox
                bbox_width = maxc - minc
                bbox_height = maxr - minr
                bbox_area = bbox_width * bbox_height
                coco_annotations["annotations"].append({
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
            for region in myelin_seg_regions:
                minr, minc, maxr, maxc = region.bbox
                bbox_width = maxc - minc
                bbox_height = maxr - minr
                bbox_area = bbox_width * bbox_height
                coco_annotations["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 2,  # Myelin category
                    "bbox": [minc, minr, bbox_width, bbox_height],
                    "area": bbox_area,
                    "iscrowd": 0,
                })
                annotation_id += 1

            image_mask_pairs.append((image_name, img))

            # Increment image_id for next image
            image_id += 1

    # Shuffle and split the dataset
    random.shuffle(image_mask_pairs)
    num_train = 7
    num_val = 2
    num_test = 1

    train_set = image_mask_pairs[:num_train]
    val_set = image_mask_pairs[num_train:num_train + num_val]
    test_set = image_mask_pairs[num_train + num_val:num_train + num_val + num_test]

    def save_set(image_mask_pairs, image_dir):
        for image_name, img in image_mask_pairs:
            cv2.imwrite(os.path.join(image_dir, image_name), img)

    save_set(train_set, train_images_dir)
    save_set(val_set, val_images_dir)
    save_set(test_set, test_images_dir)

    # Save COCO annotations
    with open(train_annotations_file, "w") as f:
        json.dump(coco_annotations, f)
    with open(val_annotations_file, "w") as f:
        json.dump(coco_annotations, f)
    with open(test_annotations_file, "w") as f:
        json.dump(coco_annotations, f)
        

if __name__ == '__main__':
    preprocess_data_yolo()
    preprocess_data_retinanet()
