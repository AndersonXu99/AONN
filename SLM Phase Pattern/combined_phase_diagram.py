# imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

### ---------------------------------------------------------------------------------------------------------------- ###
### AONN - Combined Phase Diagram                                                                                    ###
### Author: Anderson Xu                                                                                              ###
### ---------------------------------------------------------------------------------------------------------------- ###

# Defining the master file path
master_folder_path = r"C:\Users\zxq220007\Box\Quantum Optics Lab\TeTON OANN Testbed\Data 2024\May 20 2024\Save"

# open the master file path which contains all sub folders
master_folder = os.listdir(master_folder_path)

# i want to sort the master_folder variable 1, 2, 3, 4 order
master_folder.sort()

# within the master folder there are folders named from _1 to _144, create the path for each folder and read the .mat files 
data = {}

for part_folder in master_folder:
    folder_path = os.path.join(master_folder_path, part_folder)
    if os.path.isdir(folder_path):  # Check if it's a directory
        # within each folder there are .mat files, read each file
        files = os.listdir(folder_path)
        # define the most outter layer of the dictionary

        # modify the part_folder string by getting rid of the underscore and converting it to an integer
        part_folder = int(part_folder[1:])
        data[part_folder] = {}

        for file in files:
            # skip the 0.mat file
            if file != '0.mat':
                file_path = os.path.join(folder_path, file)
                # pad the data dictionary key with the folder name plus the file name
                error_file_name = file
                data[part_folder][error_file_name] = loadmat(file_path)

# a three level nested dictionary
# the keys for the most innner layer are ['__header__', '__version__', '__globals__', 'Column', 'Measured', 'Parameter', 'Pattern', 'Row', 'W_theory', 'error_mean', 'rescale']
# for example, to access a certain variable, data['_1']['error_1.mat']['Pattern']

# when the std is added to the mat file, we will use this to filter out the error file that has the least std for each sub folder
# needs more work to implement
'''
phase_data = {}
for part_folders in data.keys():
    # define the most outter layer of the dictionary
    phase_data[part_folders] = {}
    for error_file in data[part_folders].keys():
        # have a simple sorting algorithm that finds the error file with the least std
        if 'error' in error_file:
'''

# assuming that we have picked out the error file with the least std for each sub folder
# we will use the pattern field from each error file to create the combined phase diagram

# for the purpose of the demonstration, we will use the first error file from each sub folder
# create the phase_data dictionary
phase_data = {}
for part_folder in data.keys():
    # define the most outter layer of the dictionary
    phase_data[part_folder] = {}
    for error_file in data[part_folder].keys():
        # define the second layer of the dictionary
        phase_data[part_folder][error_file] = {}
        # define the third layer of the dictionary
        phase_data[part_folder][error_file] = data[part_folder][error_file]
        break

# print out the error file names 
# for debugging purposes
# for part in phase_data.keys():
#     for error_file in phase_data[part].keys():
#         print (part, error_file)

# 12 x 12 grid
Dim = np.array([12, 12])
size_real = np.array([1920, 1080]) 
size_real = size_real / Dim 

# make sure size_real is an integer
size_real = size_real.astype(int)
Pattern = np.zeros((1080, 1920), dtype=np.float64)

# now loop through the phase_data dictionary and extract the Pattern field from each error file
for part in phase_data.keys():
    for error_file in phase_data[part].keys():
        # extract the Pattern field
        
        x = int(Dim[0] - 1 - np.mod(part - 1, Dim[0]))
        y = int(np.floor((part - 1) / Dim[0]))  
        # print (x, y)
        Pattern[y*size_real[1] : (y+1)*size_real[1], x*size_real[0]:(x+1)*size_real[0]] = phase_data[part][error_file]['Pattern']

# plot the combined phase diagram
plt.imshow(Pattern, cmap='gray')

# show the plot
plt.show()