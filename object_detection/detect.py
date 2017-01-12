from __future__ import division

import sys
import cv2
import numpy as np

from matplotlib import pyplot as plt


WINDOW_SIZE = 700
GREEN = (0, 255, 0)


def show(image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image, interpolation="nearest")
    plt.show()
    return


def overlay_mask(mask, image):
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(mask_rgb, 0.5, image, 0.5, 0)


def find_biggest_contour(image):
    image_copy = image.copy()
    contours, hiearchy = cv2.findContours(image_copy, cv2.RETR_LIST,
                                          cv2.CHAIN_APPROX_SIMPLE)

    biggest = None
    biggest_area = 0
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area > biggest_area:
            biggest = contour
            biggest_area = contour_area

    mask = np.zeros(image_copy.shape, np.uint8)
    cv2.drawContours(mask, [biggest], -1, 255, -1)
    return biggest, mask


def circle_contour(image, contour):
    image_with_ellipse = image.copy()
    ellipse = cv2.fitEllipse(contour)
    cv2.ellipse(image_with_ellipse, ellipse, GREEN, 2, cv2.CV_AA)
    return image_with_ellipse


def find_pool_balls(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    scale = WINDOW_SIZE / max(image.shape)
    image = cv2.resize(image, None, fx=scale, fy=scale)

    image_blur = cv2.GaussianBlur(image, (7, 7), 0)
    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)

    min_red1 = np.array([ 0, 100,  80])
    max_red1 = np.array([10, 256, 256])

    mask1 = cv2.inRange(image_blur_hsv, min_red1, max_red1)

    min_red2 = np.array([170, 100,  80])
    max_red2 = np.array([180, 256, 256])

    mask2 = cv2.inRange(image_blur_hsv, min_red2, max_red2)

    mask = mask1 + mask2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

    big_contour, big_mask = find_biggest_contour(mask_clean)

    overlay = overlay_mask(mask_clean, image)

    circled = circle_contour(overlay, big_contour)

    return cv2.cvtColor(circled, cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    image_path = sys.argv[1]

    image = cv2.imread(image_path)
    result = find_pool_balls(image)

    show(result)
