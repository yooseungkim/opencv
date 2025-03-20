import cv2
import numpy as np


img = cv2.imread("./data/background_img.png")
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_green = (30, 100, 100)
upper_green = (100, 255, 255)
mask = cv2.inRange(hsv_img, lower_green, upper_green)

cv2.imshow("mask", mask)
cv2.waitKey(0)
