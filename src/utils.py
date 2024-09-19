import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import json
from skimage import filters, measure
from sklearn.model_selection import train_test_split

def load_bids_images(data_path):
    """Loads a BIDS-formatted dataset and index it into a dictionary"""
    data_path = Path(data_path)
    samples = pd.read_csv(data_path / 'samples.tsv', delimiter='\t')
    data_dict = {}
    for i, row in samples.iterrows():
        subject = row['participant_id']
        sample = row['sample_id']
        if subject not in data_dict:
            data_dict[subject] = {}
        data_dict[subject][sample] = {}

    sample_count = 0
    for subject in data_dict.keys():
        samples = data_dict[subject].keys()
        img_path = data_path / subject / 'micr'
        segs_path = data_path / 'derivatives' / 'labels' / subject / 'micr'
        images = list(img_path.glob('*.png'))
        axon_segs = list(segs_path.glob('*_seg-axon-*'))
        myelin_segs = list(segs_path.glob('*_seg-myelin-*'))
        for sample in samples:
            for img in images:
                if sample in str(img):
                    data_dict[subject][sample]['image'] = str(img)
                    sample_count += 1
            for axon_seg in axon_segs:
                if sample in str(axon_seg):
                    data_dict[subject][sample]['axon'] = str(axon_seg)
            for myelin_seg in myelin_segs:
                if sample in str(myelin_seg):
                    data_dict[subject][sample]['myelin'] = str(myelin_seg)
        #add pixel_size
        json_sidecar = next((data_path / subject / 'micr').glob('*.json'))
        with open(json_sidecar, 'r') as f:
            sidecar = json.load(f)
        data_dict[subject]['sidecar'] = sidecar["PixelSize"][0]

    return data_dict

def adjust_pixel_values(img, pixel_size):
    """Scales pixel values based on the provided pixel size."""
    target_pixel_size = 0.1
    scale_factor = pixel_size / target_pixel_size
    img = (img / scale_factor).astype(np.uint8)
    return img

def load_bids_image(image_path, pixel_size_metadata=None):
    """Loads a BIDS-formatted image and adjusts pixel values if needed."""
    img = cv2.imread(image_path)
    if pixel_size_metadata:
        img = adjust_pixel_values(img, pixel_size_metadata)
    return img

def resize_and_pad(img, target_size=(416, 416)):
    """Resizes an image while maintaining aspect ratio and pads to target size."""
    old_size = img.shape[:2]
    ratio = min(target_size[0] / old_size[0], target_size[1] / old_size[1])
    new_size = tuple([int(x * ratio) for x in old_size])

    img = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = target_size[1] - new_size[1]
    delta_h = target_size[0] - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img

def normalize_and_window(img):
    """Normalizes and applies intensity windowing to an image."""
    img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
    low_percentile, high_percentile = np.percentile(img, (2, 98))
    img = np.clip(img, low_percentile, high_percentile)
    img = (img - img.min()) / (img.max() - img.min())  # Renormalize
    img = (img * 255).astype(np.uint8)  # Scale to [0, 255]
    return img

def find_regions(img):
    """Finds connected regions directly from a binary segmentation mask."""
    # Ensure the image is of integer type for labeling
    img = img.astype(int)

    # Label connected regions
    labeled_img = measure.label(img)

    # Extract region properties
    regions = measure.regionprops(labeled_img)
    if len(regions) == 0:
        print("No regions found!")

    return regions

def crop_and_resize(img, region, target_size=(416, 416)):
    """Crops an image around a region and resizes it."""
    if region.area == 0:
        print("Region area is zero!")
        return np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)  # Return a black image

    minr, minc, maxr, maxc = region.bbox
    cropped_img = img[minr:maxr, minc:maxc]
    return resize_and_pad(cropped_img, target_size)

def convert_to_serializable(data):
    serializable_data = []
    for image_name, img, *rest in data:
        serializable_data.append((image_name, img.tolist(), *rest))
    return serializable_data

def shuffle_and_split(image_mask_pairs: list):
    """Shuffles data and splits."""

    train_set, test_set = train_test_split(image_mask_pairs, test_size=0.1)
    train_set, val_set = train_test_split(train_set, test_size=1/9)

    train_set_serializable = convert_to_serializable(train_set)
    val_set_serializable = convert_to_serializable(val_set)
    test_set_serializable = convert_to_serializable(test_set)

    data_split = {
        "train": train_set_serializable,
        "val": val_set_serializable,
        "test": test_set_serializable,
    }

    with open('data_split.json', 'w') as json_file:
        json.dump(data_split, json_file, indent=4)

    return train_set, test_set, val_set
