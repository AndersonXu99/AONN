import cv2
import numpy as np
import csv
import os

### ---------------------------------------------------------------------------------------------------------------- ###
### AONN - Input Intensity Finder                                                                                    ###
### Author: Anderson Xu                                                                                              ###
### ---------------------------------------------------------------------------------------------------------------- ###

# Loading the image
file_path = r"C:\Users\zxq220007\Box\Quantum Optics Lab\TeTON OANN Testbed\Data 2024\May 22 2024\Input intensity"
image_name = r"Input Intensity.png"


image_path = os.path.join(file_path, image_name)
image = cv2.imread(image_path)
if image is None:
    print("Error: Image not found.")
    exit()

# we need to rotate the image by a given angle
# we will prompt the user to draw a vertical line on the given image, and then use that line to find the angle of rotation by comparing it to the vertical axis
def draw_line(event, x, y, flags, param):
    global line
    if event == cv2.EVENT_LBUTTONDOWN:
        line.append((x, y))

cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("image", draw_line)

line = []
while True:
    image_copy = image.copy()

    if len(line) == 2:
        cv2.line(image_copy, line[0], line[1], (0, 255, 0), 1)

    cv2.imshow("image", image_copy)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# calculate the angle of rotation
# we will use the arctan function to calculate the angle of rotation
# we will use the numpy.arctan() function to calculate the angle of rotation
# the numpy.arctan() function takes in two arguments: the y-coordinate and the x-coordinate
x1, y1 = line[0]
x2, y2 = line[1]
if x2 - x1 == 0:
    angle = 0
else:
    angle = np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi  - 90

print("Angle of rotation: ", angle)

scale = 1
center = (image.shape[1] // 2, image.shape[0] // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

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

output_file_path = os.path.join(file_path, "input_intensity_boxes.txt")

# join the folder path with the file name to get the full path of the image file
# image_path = os.path.join(folder_path, image_file)

# print(rectangles)

# now that we have indicated the regions of interest, we will calculate the intensity of each region by summing the pixel values
# and then write the results to a text file
# first convert the image to grayscale
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
intensity_array = []
with open(output_file_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["Part #", "Normalized Intensity"])

    for rectangle in rectangles:
        top_left = rectangle['top_left']
        bottom_right = rectangle['bottom_right']

        if top_left or bottom_right:
            roi = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            intensity = np.sum(roi)

            intensity_array.append(intensity)
            
            # writer.writerow([top_left, bottom_right, intensity])
            # print(intensity)

        
    # we want to normalize the intensity values to the range [0, 1] using the largest intensity value
    max_intensity = max(intensity_array)
    normalized_intensity = [intensity / max_intensity for intensity in intensity_array]

    print(normalized_intensity)

    # write the normalized intensity values to the text file
    for i, rectangle in enumerate(rectangles):
        top_left = rectangle['top_left']
        bottom_right = rectangle['bottom_right']
        intensity = normalized_intensity[i]

        writer.writerow([i, intensity])