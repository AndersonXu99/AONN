import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# importing supporting files from the same directory
from dcam_live_capturing import *
from beam_locator import *
from holoeye.slmdisplaysdk import *
from GS_algorithm1 import *
from GS_algorithm2 import *

# Capture live images
dcam_capture = DcamLiveCapturing(iDevice = 0)
captured_image = dcam_capture.capture_live_images()


# Check if an image was captured
if captured_image is not None:
    print("Image captured successfully.")
    print(captured_image)

    # Display the captured image using OpenCV
    cv2.imshow("Captured Image", captured_image)
    cv2.waitKey('q')  # Wait for a key press to close the window
    cv2.destroyAllWindows()  # Close the window
else:
    print("No image captured.")

# now that the image has been captured, we want to display the image so the user can use it indicate the location of the beams
# Create CrosshairLocator instance
locator = CrosshairLocator(captured_image)

# Display image with crosshairs and allow user interaction
locator.display_image_with_crosshairs()

# Get cursor locations
cursor_locations = locator.get_cursor_locations()
# print("Cursor Locations:", cursor_locations)

locator.calculate_all_beam_locations()

print("Beam Corners:", locator.beam_corners)

# Dim = np.array([12, 12])
# Column = 5
# Row = 5
# weight = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
# interval = 50


# size_real = np.array([1920, 1080]) // Dim
# Pattern = np.zeros((1080, 1920))

# temp = np.zeros(Column * Row)
# temp[:len(weight)] = weight  # W1
# weight_shaped = np.reshape(temp, (Column, Row))

# weight_shaped = np.flipud(weight_shaped)

# if time == 0:
#     Pattern_part, phi = gsw_output(size_real, weight_shaped, interval)
# else:
#     Pattern_part, phi = gs_iteration_modified(size_real, weight_shaped, interval, Pattern_last, balance)  # after the first iteration, we have a measured weight

# x = Dim[0] - 1 - np.mod(part - 1, Dim[0])
# y = np.floor((part - 1) / Dim[0])

# Pattern[int(y*size_real[1]):int((y+1)*size_real[1]), int(x*size_real[0]):int((x+1)*size_real[0])] = Pattern_part

# if Pattern.shape != (1080, 1920):
#     Pattern = Pattern.T

# Pattern = np.mod(Pattern + Correction, 2 * np.pi)