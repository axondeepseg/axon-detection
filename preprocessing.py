import cv2
import subprocess
import os
import utils
import numpy as np
def preprocess_data(target_size=(416, 416)):
    """Preprocesses the loaded BIDS data for object detection.

    Args:
        data_dict: A dictionary containing image and segmentation paths.
        target_size: Tuple specifying the desired output image size.
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
            img = utils.resize_and_pad(img, target_size)

            # Load segmentation masks and resize (maintaining aspect ratio, potentially adding padding)
            axon_seg = cv2.imread(axon_seg_path, cv2.IMREAD_GRAYSCALE)
            myelin_seg = cv2.imread(myelin_seg_path, cv2.IMREAD_GRAYSCALE)

            axon_seg = utils.resize_and_pad(axon_seg, target_size)
            myelin_seg = utils.resize_and_pad(myelin_seg, target_size)

            # Create combined mask (you might need to adjust this depending on how you want to combine them)
            combined_mask = np.zeros_like(axon_seg)
            combined_mask[axon_seg > 0] = 1  # Axon class
            combined_mask[myelin_seg > 0] = 2  # Myelin class

            # Save preprocessed images and masks (adjust paths as needed)
            cv2.imwrite(os.path.join(processed_images_dir, f"{subject}_{sample}.png"), img)
            cv2.imwrite(os.path.join(processed_masks_dir, f"{subject}_{sample}.png"), combined_mask)

if __name__ == '__main__':
    preprocess_data()