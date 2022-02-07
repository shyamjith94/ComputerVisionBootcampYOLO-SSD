import cv2
import numpy as np
import matplotlib.pyplot as plt


def display_image(image_1, image_2, title_1="orginal image", title_2=""):
    """
    display image as subplot
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(title_1)
    plt.imshow(image_1)

    plt.subplot(1, 2, 2)
    plt.title(title_2)
    plt.imshow(image_2)
    plt.show()


# bluer image
image = cv2.imread(r"Data/bird.jpg", cv2.IMREAD_COLOR)
kernel = np.ones((5, 5)) / 25  # 5*5 25 normalize
bluer_image = cv2.filter2D(image, -1, kernel)  # -1 destination depth
display_image(image_1=image, image_2=bluer_image, title_2="bluer image")


# edge detection
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
edge_image = cv2.filter2D(gray_image, -1, kernel)
display_image(image_1=gray_image, image_2=edge_image, title_2="bluer image")

# image sharpen
image = cv2.imread(r"Data/unsharp_bird.jpg", cv2.IMREAD_COLOR)
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpen_image = cv2.filter2D(image, -1, kernel)
display_image(image_1=image, image_2=sharpen_image, title_2="bluer image")

