import cv2
import subprocess
import os
import utils


def preprocessing():

    data_repo_url = "https://github.com/axondeepseg/data_axondeepseg_sem"
    data_dir = "data_axondeepseg_sem"  # Local directory for the cloned repository
    processed_dir = "../processed/images"
    target_size = (416, 416)

    if not os.path.exists(data_dir):
        subprocess.run(["git", "clone", data_repo_url])

    data_dict = utils.load_bids_images(data_dir)
    print(data_dict)



if __name__ == '__main__':
    preprocessing()