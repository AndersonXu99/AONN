import cv2
import numpy as np
import os
import csv

### ---------------------------------------------------------------------------------------------------------------- ###
### AONN - Input Intensity Finder                                                                                    ###
### Author: Anderson Xu                                                                                              ###
### ---------------------------------------------------------------------------------------------------------------- ###

# Loading the image
image_path = r"C:\Users\zxq220007\Desktop\Anderson\AONN\EIT\UniformInput.jpg"
image = cv2.imread(image_path)
if image is None:
    print("Error: Image not found.")
    exit()

# Global variables to store circle parameters
rectangles = []

# Callback function for mouse events
def draw_rectangle(event, x, y, flags, param):
    global rectangles

    if event == cv2.EVENT_LBUTTONDOWN:
        rectangles.append({'top_left': (x, y), 'bottom_right': None})

    elif event == cv2.EVENT_LBUTTONUP:
        if rectangles:
            rectangles[-1]['bottom_right'] = (x, y)

# Undo function
def undo_last_rectangle():
    global rectangles
    if rectangles:
        rectangles.pop()

cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("image", draw_rectangle)

while True:
    image_copy = image.copy()

    for rectangle in rectangles:
        if rectangle['bottom_right']:
            cv2.rectangle(image_copy, rectangle['top_left'], rectangle['bottom_right'], (0, 255, 0), 1)

    cv2.imshow("image", image_copy)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('u'):
        undo_last_rectangle()

output_file_path = "circle_parameters.txt"

print(rectangles)