import cv2
import numpy as np

def load_bids_image(image_path, pixel_size_metadata=None):
    """Loads a BIDS-formatted image and adjusts pixel values if needed."""
    img = cv2.imread(image_path)
    if pixel_size_metadata:
        img = adjust_pixel_values(img, pixel_size_metadata)
    return img

def adjust_pixel_values(img, pixel_size):
    """Scales pixel values based on the provided pixel size."""
    target_pixel_size = 0.1
    scale_factor = pixel_size / target_pixel_size
    img = (img / scale_factor).astype(np.uint8)
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
