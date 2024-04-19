import cv2
import os
import csv
import torch
import numpy as np
import concurrent.futures

### ---------------------------------------------------------------------------------------------------------------- ###
### AONN - Classification                                                                                            ###
### Author: Anderson Xu                                                                                              ###
### ---------------------------------------------------------------------------------------------------------------- ###

# Given a greyscale image with ten bright dots, the user will find the ROIs in a different function and store the information in a different file
# we want to first get all the file paths needed 
master_file_path = r""

### ---------------------------------------------------------------------------------------------------------------- ###
### Function Definitions                                                                                             ###
### ---------------------------------------------------------------------------------------------------------------- ###

# Function to load image
def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print("Error: Image not found.")
        exit()
    return image


### ---------------------------------------------------------------------------------------------------------------- ###
### Main                                                                                                             ###
### ---------------------------------------------------------------------------------------------------------------- ###

# loading the images in using threadpool
image_path = os.path.join(master_file_path, "ROI.tif")
with concurrent.futures.ThreadPoolExecutor() as executor:
    image = executor.submit(load_image, image_path)

    