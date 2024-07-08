import cv2
import subprocess
import os
import utils
import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
def preprocess_data():
    """Preprocesses the loaded BIDS data for object detection.

    Steps:
    1. Load
    2. Normalize + windowing
    3. find regions axons Use skimage.measures.regionprops()
    4. Croping + Resizing
    """
    data_repo_url = "https://github.com/axondeepseg/data_axondeepseg_sem"
    data_dir = "data_axondeepseg_sem"  # Local directory for the cloned repository
    processed_images_dir = "processed_images"
    processed_masks_dir = "processed_masks"
    target_size = (416, 416)

    if not os.path.exists(data_dir):
        subprocess.run(["git", "clone", data_repo_url])

    os.makedirs(processed_images_dir, exist_ok=True)
    os.makedirs(processed_masks_dir, exist_ok=True)

    data_dict = utils.load_bids_images(data_dir)
    bbox_data = []
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

            # Load and preprocess the images and the segmentation mask
            img = utils.load_bids_image(img_path, pixel_size)
            img = utils.normalize_and_window(img)

            # Load segmentation masks and resize (maintaining aspect ratio, potentially adding padding)
            # Load segmentation masks and find regions
            axon_seg = cv2.imread(axon_seg_path, cv2.IMREAD_GRAYSCALE)
            axon_seg_regions = utils.find_regions(axon_seg)
            for i, region in enumerate(axon_seg_regions):
                minr, minc, maxr, maxc = region.bbox
                cv2.rectangle(img, (minc, minr), (maxc, maxr), (255, 0, 0), 2)  # Blue rectangle for axon
                bbox_data.append({"image_name": f"{subject}_{sample}.png", "xmin": minc, "ymin": minr, "xmax": maxc, "ymax": maxr, "class": "axon"})

            # Process Myelin
            myelin_seg = cv2.imread(myelin_seg_path, cv2.IMREAD_GRAYSCALE)
            myelin_seg_regions = utils.find_regions(myelin_seg)
            for i, region in enumerate(myelin_seg_regions):
                minr, minc, maxr, maxc = region.bbox
                cv2.rectangle(img, (minc, minr), (maxc, maxr), (0, 255, 0), 2)  # Green rectangle for myelin
                bbox_data.append({"image_name": f"{subject}_{sample}.png", "xmin": minc, "ymin": minr, "xmax": maxr, "ymax": maxc, "class": "myelin"})

            # Save the image with bounding boxes
            image_name = f"{subject}_{sample}.png"
            cv2.imwrite(os.path.join(processed_images_dir, image_name), img)

def visualize_processed_images():
    # Path to your processed images directory
    processed_images_dir = "processed_images"

    # Load the bounding box annotations
    annotations = pd.read_csv(os.path.join(processed_images_dir, "annotations.csv"))

    # Iterate through images and draw bounding boxes
    for index, row in annotations.iterrows():
        image_path = os.path.join(processed_images_dir, row["image_name"])
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error loading image: {image_path}")
            continue

        # Extract bounding box coordinates
        xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]

        # Draw bounding box on image
        color = (0, 255, 0) if row["class"] == "myelin" else (255, 0, 0)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)

        # Display image
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for matplotlib
        plt.title(row["image_name"])
        plt.show()

if __name__ == '__main__':
    preprocess_data()
    visualize_processed_images()