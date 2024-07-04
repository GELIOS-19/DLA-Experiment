import numpy as np
import cv2
import matplotlib.pyplot as plt


def pseudo_erosion(image, iterations):
    """
    Perform iterative pseudo-erosion on the input image.

    Parameters:
    image (numpy.ndarray): Input grayscale image.
    iterations (int): Number of iterations to perform.

    Returns:
    numpy.ndarray: Eroded image.
    """
    eroded_image = cv2.erode(image, None, iterations=iterations)
    return eroded_image


# Load the input image
image_path = 'invertedheightmap.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Define the number of iterations
iterations = 0

# Perform pseudo-erosion
eroded_image = pseudo_erosion(image, iterations)


# Display the original and eroded images
plt.figure(figsize=(5, 5))
# plt.subplot(1, 2, 1)
# plt.title('Original Image')
# plt.imshow(image, cmap='gray')
# plt.subplot(1, 2, 2)
plt.title(f'Eroded Image ({iterations} iterations)')
plt.imshow(eroded_image, cmap='gray')
plt.show()
