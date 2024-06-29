import data_loader
import cv2
import numpy as np
import matplotlib.pyplot as plt


def preprocessing():
    path = 'downloaded_image.png'
    data = data_loader.loadImage()
    image = cv2.imread(path)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Perform edge detection using Canny
    edges = cv2.Canny(blurred_image, 50, 150)

    # Display the results
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Grayscale Image')
    plt.imshow(gray_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Edges')
    plt.imshow(edges, cmap='gray')
    plt.axis('off')

    plt.show()

if __name__ == '__main__':
    preprocessing()