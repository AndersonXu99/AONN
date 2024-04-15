import cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import natsort
import concurrent.futures
import time

start_time = time.time()

### ---------------------------------------------------------------------------------------------------------------- ###
### AONN                                                                                                             ###
### Author: Anderson Xu                                                                                              ###
### ---------------------------------------------------------------------------------------------------------------- ###



### ---------------------------------------------------------------------------------------------------------------- ###
### User Input Section                                                                                               ###
### ---------------------------------------------------------------------------------------------------------------- ###
# Please enter the file path to the MAIN folder and also name the file that you would like the results to output to
# For example: C:\Users\zxq220007\Box\Quantum Optics Lab\TeTON OANN Testbed\Data 2024\Apr 11 2024\138C\5X5 Trans5 50mm cell 138C 290MHz 3037MHz
# No need to double lashes in the file path
# Important: Keep the r in front of the quotation mark
data_folder_path = r"C:\Users\zxq220007\Box\Quantum Optics Lab\TeTON OANN Testbed\Data 2024\Apr 15 2024\5X5 Trans5 50mm cell 138C Onres 3037MHz"
output_file_name = "Result"

# path to the csv that maps the 0-20 values to actual powers of the beam
csv_file_path = r'C:\Users\zxq220007\Box\Quantum Optics Lab\TeTON OANN Testbed\Data 2024\Apr 11 2024\5X5 Mod Depth to Power updated - Copy.csv'
### ---------------------------------------------------------------------------------------------------------------- ###



### ---------------------------------------------------------------------------------------------------------------- ###
### Function Definitions                                                                                             ###
### ---------------------------------------------------------------------------------------------------------------- ###
def calculate_EIT(image1, roi1):
    # Create the first ROI using numpy slicing
    roi1_gray = image1[roi1[0]:roi1[1], roi1[2]:roi1[3]]
    # Calculate the sum of grayscale intensity within the first ROI
    intensity_EIT = np.sum(roi1_gray, dtype=float)
    return intensity_EIT

def calculate_background(image2, roi2):
    # Create the first ROI using numpy slicing
    roi2_gray = image2[roi2[0]:roi2[1], roi2[2]:roi2[3]]
    # Calculate the sum of grayscale intensity within the first ROI
    intensity_background = np.sum(roi2_gray, dtype=float)

    return intensity_background

def draw_rois_on_image(image, rois, output_folder, file_name):
    # Create a copy of the image to avoid modifying the original
    image_with_rois = image.copy()

    # Draw each ROI rectangle on the image
    for roi in rois:
        cv2.rectangle(image_with_rois, (roi[2], roi[0]), (roi[3], roi[1]), (0, 255, 0), 1)  # Green rectangle

    # Save the image with the drawn ROIs
    output_path = os.path.join(output_folder, f'{file_name}.png')
    cv2.imwrite(output_path, image_with_rois)

# Function to load an image given its file path
def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_COLOR)

def rename_files(original_name, desired_name, BG_folder_path, EIT_folder_path):
    # for when power is 0, buffer the name to be 0.00
    if desired_name == '0':
        desired_name = '0.00'
    # Construct full paths for the original and desired files
    BG_original_path = os.path.join(BG_folder_path, original_name + '.TIFF')
    BG_desired_path = os.path.join(BG_folder_path, desired_name + '.TIFF')

    # Construct full paths for the original and desired files
    EIT_original_path = os.path.join(EIT_folder_path, original_name + '.TIFF')
    EIT_desired_path = os.path.join(EIT_folder_path, desired_name + '.TIFF')

    # Rename the file
    try:
        os.rename(BG_original_path, BG_desired_path)
        os.rename(EIT_original_path, EIT_desired_path)
    except FileNotFoundError:
        print(f'File {BG_original_path} not found.')
### ---------------------------------------------------------------------------------------------------------------- ###

### ---------------------------------------------------------------------------------------------------------------- ###
### File Handling and Image Loading                                                                                  ###
### ---------------------------------------------------------------------------------------------------------------- ###
# Defining all file paths needed
EIT_main_folder = os.path.join(data_folder_path, 'EIT')  # Directory for the first set of images
BG_main_folder = os.path.join(data_folder_path, 'BG')  # Directory for the second set of images
roi_image_dir = os.path.join(data_folder_path, 'ROI.tif')  # ROI image path
main_output_folder = os.path.join(data_folder_path, output_file_name)
os.makedirs(main_output_folder, exist_ok=True)  # Create a main output folder

# Renaming the files
with open(csv_file_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)  # Skip header row

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for row in csv_reader:
            executor.submit(rename_files, row[0], row[2], BG_main_folder, EIT_main_folder)
print('Renaming process complete.')
end_time = time.time()

# Get the list of files in the image directories
image_files_1 = os.listdir(EIT_main_folder)
image_files_2 = os.listdir(BG_main_folder)

# Sort the file lists to ensure consistent pairing
image_files_1 = sorted(image_files_1)
image_files_2 = sorted(image_files_2)

# reading the reference ROI image
ROI_image = cv2.imread(roi_image_dir, cv2.IMREAD_COLOR)

# Combine directory paths with filenames in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    image_paths_1 = list(executor.map(lambda filename: os.path.join(EIT_main_folder, filename), image_files_1))
    image_paths_2 = list(executor.map(lambda filename: os.path.join(BG_main_folder, filename), image_files_2))

print("Loading images")
# Load images in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    images_1 = list(executor.map(load_image, image_paths_1))
    images_2 = list(executor.map(load_image, image_paths_2))
print("Images Loaded")
### ---------------------------------------------------------------------------------------------------------------- ###



### ---------------------------------------------------------------------------------------------------------------- ###
### Initialization                                                                                                   ###
### ---------------------------------------------------------------------------------------------------------------- ###
# Define the ROI coordinates (top, bottom, left, right) for 20 ROIs
roi_coords = [(239, 272, 318, 348), (267, 295, 680, 716), (282, 315, 1068, 1100), (334, 365, 1440, 1476),
              (375, 409, 1813, 1853),
              (595, 624, 272, 306), (614, 648, 648, 679), (662, 688, 1025, 1056), (694, 725, 1392, 1433),
              (730, 757, 1779, 1818),
              (941, 970, 240, 281), (968, 1004, 616, 649), (1024, 1069, 988, 1032), (1047, 1079, 1365, 1399),
              (1088, 1120, 1748, 1778),
              (1300, 1326, 201, 238), (1331, 1364, 578, 615), (1371, 1410, 965, 1008), (1398, 1425, 1339, 1372),
              (1445, 1477, 1711, 1748),
              (1651, 1682, 172, 206), (1689, 1725, 549, 585), (1735, 1763, 924, 964), (1775, 1803, 1317, 1356),
              (1818, 1845, 1686, 1720)]  # Example coordinates for 25 ROIs

# Initialize lists to store EIT and Background values for each ROI
EIT_values = []
intensity_difference_values = []

# the beam intensities
file_names_beam = []
### ---------------------------------------------------------------------------------------------------------------- ###



### ---------------------------------------------------------------------------------------------------------------- ###
### Main Function                                                                                                    ###
### ---------------------------------------------------------------------------------------------------------------- ###
# mark roi image
draw_rois_on_image(ROI_image, roi_coords, main_output_folder, f'ROI_All')

# process to find the array of power of each beam
for file1, file2 in zip(natsort.natsorted(image_files_1), natsort.natsorted(image_files_2)):
    # Extract the image numbers from the filenames
    image_num1 = float(os.path.splitext(file1)[0])
    file_names_beam.append(image_num1)

# Process each Beam separately
for beam_index, (roi1, roi2) in enumerate(zip(roi_coords, roi_coords)):
    # Initialize lists to store file names and intensity differences for each Beam
    intensity_differences_beam = []
    EIT_values_beam = []
    background_values_beam = []

    # Create a subfolder for each Beam inside the main output folder
    beam_folder = os.path.join(main_output_folder, f'Beam{beam_index + 1}')
    os.makedirs(beam_folder, exist_ok=True)

    # Process the image pairs
    for image1, image2 in zip(images_1, images_2):
        # Calculate the intensity difference for each Beam and accumulate the results
        EIT_value = calculate_EIT(image1, roi1)
        background_value = calculate_background(image2, roi2)
        intensity_diff = EIT_value - background_value

        # storing the values
        EIT_values_beam.append(EIT_value)
        background_values_beam.append(background_value)
        intensity_differences_beam.append(intensity_diff)

    # storing the values for the final overall plot
    intensity_difference_values.append(intensity_differences_beam)
    EIT_values.extend(EIT_values_beam)
    # Create a CSV file in the subfolder to store the intensity differences for each Beam
    csv_file_path_beam = os.path.join(beam_folder, f'EIT_Intensity_Beam{beam_index + 1}.csv')
    with open(csv_file_path_beam, 'w', newline='') as csv_file_beam:
        csv_writer_beam = csv.writer(csv_file_beam)
        csv_writer_beam.writerow(['File', 'EIT_Intensity'])

        # Write the image number and intensity difference to the CSV file for each Beam
        for file_name, intensity_diff_beam in zip(file_names_beam, intensity_differences_beam):
            csv_writer_beam.writerow([file_name, intensity_diff_beam])

    # Intensity curve of each individual beam -
    # plt.figure()
    # Plot a graph with normalized data and fitted curve for each Beam
    # plt.plot(file_names_beam, intensity_differences_beam, marker='o', label=f'Intensity curve - Beam{beam_index + 1}')
    # plt.xlabel('Power in mW')
    # plt.xticks([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000])
    # plt.ylabel('Intensity')
    # plt.title(f'Intensity vs Power - Beam{beam_index + 1}')
    # plt.grid(True)
    # plt.legend()

    # Save the plot to the subfolder for each Beam
    # plot_file_path_beam = os.path.join(beam_folder, f'Intensity vs Power_Beam{beam_index + 1}.TIFF')
    # plt.savefig(plot_file_path_beam)
    # plt.close()

    # Show the plot for each Beam
    # plt.show()

    # Plot the graph with EIT and Background values for each Beam
    plt.figure()  # Create a new figure for each Beam
    plt.plot(file_names_beam, EIT_values_beam, marker='x', label=f'EIT Beam {beam_index + 1}')
    plt.plot(file_names_beam, background_values_beam, marker='x', label=f'Background Beam {beam_index + 1}')
    plt.xlabel('Power in mW')
    # plt.xticks([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000])
    plt.ylabel('Intensity')
    plt.title(f'Background & Intensity vs Power - Beam{beam_index + 1}')
    plt.grid(True)
    plt.legend()

    # Save the plot for each Beam in its corresponding folder
    plot_file_path_beam = os.path.join(main_output_folder, f'Beam{beam_index + 1}',
                                       f'Background & Intensity vs Power_Beam{beam_index + 1}.png')
    os.makedirs(os.path.dirname(plot_file_path_beam), exist_ok=True)
    plt.savefig(plot_file_path_beam)
    plt.close()
    # plt.show()

# Plotting the overall plot
fig, axs = plt.subplots(5, 5, figsize=(12, 8))  # Create a 5x5 grid of subplots

# Iterate through each beam and plot its intensity difference curve in the corresponding subplot
for beam_index in range(len(intensity_difference_values)):
    row_index = beam_index // 5  # Calculate the row index in the grid
    col_index = beam_index % 5  # Calculate the column index in the grid
    axs[row_index, col_index].plot(file_names_beam, intensity_difference_values[beam_index], marker='o')
    axs[row_index, col_index].set_title(f'Beam {beam_index + 1}')
    # axs[row_index, col_index].set_xlabel('Power in mW')
    # axs[row_index, col_index].set_ylabel('Intensity Difference')
    axs[row_index, col_index].grid(True)

# Adjust layout to add more space between subplots
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)

# Save the overall plot to a file
overall_plot_file_path = os.path.join(main_output_folder, 'Overall_Intensity_vs_Power_Grid.png')
plt.savefig(overall_plot_file_path)

# Show the overall plot
plt.show()
plt.close()

end_time = time.time()
time_elapsed = end_time - start_time
print(f'Time elapsed: {time_elapsed}')
