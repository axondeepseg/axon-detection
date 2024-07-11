# Axon and Myelin Segmentation from Microscopy Images using Object Detection

This project focuses on preprocessing and segmenting axons and myelin from microscopy images stored in the BIDS format. It leverages OpenCV for image manipulation and is designed to work with object detection models like YOLO or RetinaNet.

## Features

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

### Running preprocessing script
   ```bash
   cd src
   ```bash
   python  preprocessing.py


   

