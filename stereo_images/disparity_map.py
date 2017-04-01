import numpy as np
import cv2
from matplotlib import pyplot as plt

LEFT = "data/NTSD-200/fluorescent/left/frame_1.png"
RIGHT = "data/NTSD-200/fluorescent/right/frame_1.png"

imgL = cv2.imread(LEFT, 0)
imgR = cv2.imread(RIGHT, 0)

stereo = cv2.StereoSGBM(1, numDisparities=64, SADWindowSize=3)
disparity = stereo.compute(imgL, imgR)

# plt.imshow(disparity, 'gray')
# plt.show()
