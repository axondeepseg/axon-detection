import cv2
import subprocess
import os
import utils
import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
def preprocess_data(target_size=(416, 416)):
    """Preprocesses the loaded BIDS data for object detection.

    Args:
        data_dict: A dictionary containing image and segmentation paths.
        target_size: Tuple specifying the desired output image size.
    Steps:
    1. Load
    2. Normalize + windowing
    3. Threshold + find regions axons Use skimage.measures.regionprops()
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

            axon_seg_regions = utils.threshold_and_find_regions(cv2.imread(axon_seg_path, cv2.IMREAD_GRAYSCALE))
            for i, region in enumerate(axon_seg_regions):
                cropped_img = utils.crop_and_resize(img, region, target_size)
                image_name = f"{subject}_{sample}_axon_{i}.png"
                cv2.imwrite(os.path.join(processed_images_dir, image_name), cropped_img)
                bbox_data.append({"image_name": image_name, "xmin": region.bbox[1], "ymin": region.bbox[0], "xmax": region.bbox[3], "ymax": region.bbox[2], "class":"axon"})

            #Process Myelin
            myelin_seg_regions = utils.threshold_and_find_regions(
                cv2.imread(myelin_seg_path, cv2.IMREAD_GRAYSCALE)
            )
            for i, region in enumerate(myelin_seg_regions):
                cropped_img = utils.crop_and_resize(img, region, target_size)

                image_name = f"{subject}_{sample}_myelin_{i}.png"
                cv2.imwrite(os.path.join(processed_images_dir, image_name), cropped_img)
                bbox_data.append({
                    "image_name": image_name,
                    "xmin": region.bbox[1],
                    "ymin": region.bbox[0],
                    "xmax": region.bbox[3],
                    "ymax": region.bbox[2],
                    "class": "myelin"
                })

        df = pd.DataFrame(bbox_data)
        df.to_csv(os.path.join(processed_images_dir, 'annotations.csv'), index=False)

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
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Green rectangle

        # Display image
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for matplotlib
        plt.title(row["image_name"])
        plt.show()

if __name__ == '__main__':
    preprocess_data()
    visualize_processed_images()