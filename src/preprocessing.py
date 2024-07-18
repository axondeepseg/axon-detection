import random
import subprocess
import utils
import cv2
import os
import shutil

def preprocess_data():
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

if __name__ == '__main__':
    preprocess_data()
