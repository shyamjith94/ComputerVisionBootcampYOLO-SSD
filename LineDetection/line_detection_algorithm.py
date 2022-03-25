import cv2
import numpy
import numpy as np
import matplotlib.pyplot as plt
import os


def draw_the_line(image, lines):
    """
    (255, 0, 0) -> BGR
    lines_image - > all zeros means black image

    :param image:
    :param lines: detected line from hough transform
    :return:
    """
    lines_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    # there are x,y coordinate for starting and end point of the line
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), thickness=3)
    # finally merge the lines and image
    image_with_lines = cv2.addWeighted(image, 0.8, lines_image, 1, 0.0)
    return image_with_lines


def region_of_interest(image, region_int_v):
    """
    cropping image to triangle(polygon) direction based region of interest value
    we are using logical AND operator
    we are going to replace pixel with 0 (black) - region we are not interested
    :return: masked image
    """
    mask = np.zeros_like(image)
    # region we interested in the lower triangle - 255 white pixels
    cv2.fillPoly(mask, region_int_v, 255)
    # we have to keep the region of original image where the shite colored pixel
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def get_detected_line(image: numpy.array([])):
    """
    Detecting image line from video frame
    Since edge detection is susceptible to noise in the image,
    first step is to remove the noise in the image with a 5x5 Gaussian filter.

    we are using canny algorithm
    https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
    """
    (height, width) = (image.shape[0], image.shape[1])
    # image to gray scale
    g_im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edge detection algorithm (Canny's algorithm')
    canny_image = cv2.Canny(g_im, 100, 120)
    # we are interested in the "lower region" of the image (there are driving lanes)
    region_of_interest_vertices = [
        (0, height),  # bottom left corner point
        (width / 2, height * 0.65),  # middle point
        (width, height)  # bottom right corner point
    ]
    # we can get rid of un-relevant part of image
    # we can just keep the lower triangle region
    cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], dtype=np.int32))
    # use line detection algorithm (use radians instead of degree, 1 degree = pi/180)
    lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi / 180, threshold=50, lines=np.array([]), minLineLength=40,
                            maxLineGap=150)
    # draw the line to image
    image_with_lines = draw_the_line(image, lines)
    return image_with_lines


if __name__ == '__main__':
    video_file = r"Data\lane_detection_video.mp4"
    os.chdir('../')
    current_dir = os.getcwd()
    image_path = os.path.join(current_dir, video_file)

    video = cv2.VideoCapture(image_path)
    while video.isOpened():
        is_grabbed, frame = video.read()

        # because the end of video
        if not is_grabbed:
            break
        image_gray = get_detected_line(frame)
        cv2.imshow('line detection video', image_gray)
        cv2.waitKey(100)

    video.release()
    cv2.destroyAllWindows()
