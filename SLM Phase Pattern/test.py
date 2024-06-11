import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# importing supporting files from the same directory
from dcam_live_capturing import *


# Capture live images
dcam_capture = DcamLiveCapturing(iDevice = 0)
captured_image = dcam_capture.capture_live_images()

# Check if an image was captured
if captured_image is not None:
    print("Image captured successfully.")
    print(captured_image)

    # Create a resizable window
    cv2.namedWindow("Captured Image", cv2.WINDOW_NORMAL)

    # Display the captured image using OpenCV
    cv2.imshow("Captured Image", captured_image)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Captured Image", cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()  # Close the window
else:
    print("No image captured.")
