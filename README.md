# Axon Detection in Microscopy Images

This project focuses on preprocessing and segmenting axons and myelin from microscopy images stored in the BIDS format. It leverages OpenCV for image manipulation and is designed to work with object detection models like YOLO or RetinaNet.

## Overview
### General:
* Loads BIDS-formatted microscopy images.
* Automatically adjusts pixel values based on metadata.
* Resizes and pads images to fit object detection model input sizes.
* Addinded
* Includes a `utils.py` file with helper functions for easy preprocessing.


## Getting Started

### Prerequisites

* Python 3.x
* OpenCV (`pip install opencv-python`)

### Installation

1. **Create a Virtual Environment:**

   ```bash
   python -m venv venv
   
2. **Activate a Virtual Environment:**

   ```bash
   source venv/bin/activate
   
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

### Running scripts
   ```bash
### Old yolo model:
   cd src
   python preprocessing.py
   python yolo_train.py

### Specific script for RetinaNet model:
   ```bash
   cd src
   python -m retinaNet.retinaNet_train
