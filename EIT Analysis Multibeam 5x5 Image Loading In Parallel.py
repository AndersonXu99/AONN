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
### OANN                                                                                                             ###
### Authoer: Anderson Xu                                                                                             ###
### ---------------------------------------------------------------------------------------------------------------- ###

### ---------------------------------------------------------------------------------------------------------------- ###
### User Input Section                                                                                               ###
### ---------------------------------------------------------------------------------------------------------------- ###
# Please enter the file path to the MAIN folder and also name the file that you would like the results to output to
# For example: C:\Users\zxq220007\Box\Quantum Optics Lab\TeTON OANN Testbed\Data 2024\Apr 11 2024\138C\5X5 Trans5 50mm cell 138C 290MHz 3037MHz
# No need to double lashes in the file path
data_folder_path = "C:\Users\zxq220007\Box\Quantum Optics Lab\TeTON OANN Testbed\Data 2024\Apr 11 2024\138C\5X5 Trans5 50mm cell 138C 290MHz 3037MHz"
output_file_name = "Insert here - test again"  # Keep the slash before your title
csv_file_path = r'C:\Users\zxq220007\Box\Quantum Optics Lab\TeTON OANN Testbed\Data 2024\Apr 11 2024\2X2 Mod Depth to Power updated.csv'
### ---------------------------------------------------------------------------------------------------------------- ###



### ---------------------------------------------------------------------------------------------------------------- ###
### Function Definitions                                                                                             ###
### ---------------------------------------------------------------------------------------------------------------- ###
def calculate_EIT(image1, roi1):
    # Create the first ROI using numpy slicing
    roi1_gray = image1[roi1[0]:roi1[1], roi1[2]:roi1[3]]
    # Calculate the sum of grayscale intensity within the first ROI
    intensity_EIT = np.sum(roi1_gray)
    return intensity_EIT

def calculate_background(image2, roi2):
    # Create the first ROI using numpy slicing
    roi2_gray = image2[roi2[0]:roi2[1], roi2[2]:roi2[3]]

    # Calculate the sum of grayscale intensity within the first ROI
    intensity_background = np.sum(roi2_gray)

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
### ---------------------------------------------------------------------------------------------------------------- ###



### ---------------------------------------------------------------------------------------------------------------- ###
### File Handling and Image Loading                                                                                  ###
### ---------------------------------------------------------------------------------------------------------------- ###
# Defining all file paths needed
EIT_main_folder = os.path.join(data_folder_path, 'EIT')  # Directory for the first set of images
BG_main_folder = os.path.join(data_folder_path, 'BG')  # Directory for the second set of images
image_dir_roi = os.path.join(data_folder_path, 'ROI.tiff')  # ROI image path
os.makedirs(os.path.join(data_folder_path, output_file_name), exist_ok=True) # Create a main output folder

# Get the list of files in the image directories
image_files_1 = os.listdir(EIT_main_folder)
image_files_2 = os.listdir(BG_main_folder)

# Sort the file lists to ensure consistent pairing
image_files_1 = natsort.natsorted(image_files_1)
image_files_2 = natsort.natsorted(image_files_2)

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


# Define the ROI coordinates (top, bottom, left, right) for 20 ROIs
roi_coords = [(234, 260, 359, 391), (254, 280, 724, 753), (277, 306, 1109, 1132), (310, 337, 1472, 1499),
              (347, 369, 1839, 1869),
              (587, 610, 321, 344), (600, 631, 702, 727), (630, 650, 1074, 1094), (661, 692, 1430, 1467),
              (698, 726, 1809, 1838),
              (931, 995, 299, 327), (960, 989, 660, 694), (980, 1015, 1020, 1051), (1020, 1046, 1406, 1437),
              (1056, 1081, 1773, 1818),
              (1286, 1310, 260, 288), (1318, 1341, 634, 663), (1340, 1367, 1000, 1035), (1380, 1405, 1369, 1422),
              (1410, 1435, 1753, 1780),
              (1654, 1676, 229, 258), (1681, 1703, 598, 627), (1715, 1736, 975, 1010), (1741, 1766, 1349, 1382),
              (1774, 1798, 1726, 1759)]  # Example coordinates for 25 ROIs





# Sort the file lists to ensure consistent pairing
# image_files_1.sort()
# image_files_2.sort()

# Initialize lists to store EIT and Background values for each ROI
EIT_values = []
intensity_difference_values = []

imageroi = cv2.imread(image_dir_roi, cv2.IMREAD_COLOR)

# mark roi image
draw_rois_on_image(imageroi, roi_coords, output_file_name, f'ROI_All')

# process to find the array of power of each beam
file_names_beam = []
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
