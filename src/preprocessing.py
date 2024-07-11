import subprocess
import utils
import cv2
import os
def preprocess_data():
    """Preprocesses the loaded BIDS data for object detection.

    Steps:
    1. Load
    2. Normalize + windowing
    3. find regions axons Use skimage.measures.regionprops()
    4. Croping + Resizing
    """

    # TODO: Issue of raw path
    data_repo_url = "https://github.com/axondeepseg/data_axondeepseg_sem"
    data_dir = "data_axondeepseg_sem"  # Local directory for the cloned repository
    processed_images_dir = "processed_images"
    processed_masks_dir = "processed_masks"

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

if __name__ == '__main__':
    preprocess_data()
