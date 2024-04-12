import cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import natsort

# Function to calculate intensity difference between two ROIs
def calculate_intensity_difference(image1, image2, roi1, roi2):
    # Create the first ROI using numpy slicing
    roi1_gray = image1[roi1[0]:roi1[1], roi1[2]:roi1[3]]

    # Create the second ROI using numpy slicing
    roi2_gray = image2[roi2[0]:roi2[1], roi2[2]:roi2[3]]

    # Calculate the sum of grayscale intensity within the first ROI
    intensity_sum1 = np.sum(roi1_gray)

    # Calculate the sum of grayscale intensity within the second ROI
    intensity_sum2 = np.sum(roi2_gray)

    # Subtract the intensities of the second ROI from the first ROI
    intensity_diff = intensity_sum1 - intensity_sum2

    return intensity_diff

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

# Define the directories containing the image pairs
image_dir_1 = 'C:\\Users\\zxq220007\\Box\\Quantum Optics Lab\\TeTON OANN Testbed\\Data 2024\\Apr 11 2024\\138C\\2X2 Trans5 50mm cell 138C Onres 3036MHz\\EIT'  # Directory for the first set of images
image_dir_2 = 'C:\\Users\\zxq220007\\Box\\Quantum Optics Lab\\TeTON OANN Testbed\\Data 2024\\Apr 11 2024\\138C\\2X2 Trans5 50mm cell 138C Onres 3036MHz\\BG'  # Directory for the second set of images
image_dir_roi = 'C:\\Users\\zxq220007\\Box\\Quantum Optics Lab\\TeTON OANN Testbed\\Data 2024\\Apr 11 2024\\138C\\2X2 Trans5 50mm cell 138C Onres 3036MHz\\ROI.tiff' #ROI image path
# Define the ROI coordinates (top, bottom, left, right) for 20 ROIs
roi_coords = [(436,470,526,578),(524,551,1604,1653),(964,989,1052,1091),
              (1459,1487,442,487),(1543,1574,1529,1570)]   # Example coordinates for 25 ROIs

# Get the list of files in the image directories
image_files_1 = os.listdir(image_dir_1)
image_files_2 = os.listdir(image_dir_2)

# Sort the file lists to ensure consistent pairing
image_files_1.sort()
image_files_2.sort()

# Create a main output folder only once
main_output_folder = 'C:\\Users\\zxq220007\\Box\\Quantum Optics Lab\\TeTON OANN Testbed\\Data 2024\\Apr 11 2024\\138C\\2X2 Trans5 50mm cell 138C Onres 3036MHz\\EIT v Power 1'
os.makedirs(main_output_folder, exist_ok=True)

# Initialize lists to store EIT and Background values for each ROI
EIT_values = [[] for _ in range(len(roi_coords))]
background_values = [[] for _ in range(len(roi_coords))]

imageroi = cv2.imread(image_dir_roi, cv2.IMREAD_COLOR)

#mark roi image
draw_rois_on_image(imageroi, roi_coords, main_output_folder, f'ROI_All')

# Process each Beam separately
for beam_index, (beam1, beam2) in enumerate(zip(roi_coords, roi_coords)):
    # Create a subfolder for each Beam inside the main output folder
    beam_folder = os.path.join(main_output_folder, f'Beam{beam_index + 1}')
    os.makedirs(beam_folder, exist_ok=True)

    # Initialize lists to store file names and intensity differences for each Beam
    file_names_beam = []
    intensity_differences_beam = []

    # Initialize lists to store EIT and Background values for each Beam
    EIT_values_beam = []
    background_values_beam = []

    # Process the image pairs
    for file1, file2 in zip(natsort.natsorted(image_files_1), natsort.natsorted(image_files_2)):
        # Extract the image numbers from the filenames
        image_num1 = float(os.path.splitext(file1)[0])

        # Construct the file paths for the image pairs
        image_path1 = os.path.join(image_dir_1, file1)
        image_path2 = os.path.join(image_dir_2, file2)

        # Load the U16 images
        image1 = cv2.imread(image_path1, cv2.IMREAD_COLOR)
        image2 = cv2.imread(image_path2, cv2.IMREAD_COLOR)

        # Calculate the intensity difference for each Beam and accumulate the results
        intensity_diff = calculate_intensity_difference(image1, image2, beam1, beam2)

        # Calculate EIT and Background values
        EIT_value = calculate_EIT(image1, beam1)
        background_value = calculate_background(image2, beam2)
        EIT_values_beam.append(EIT_value)
        background_values_beam.append(background_value)

        # Append the image number and intensity difference to the lists for each Beam
        file_names_beam.append(image_num1)
        intensity_differences_beam.append(intensity_diff)

        # Draw and save the image with the Beam for each Beam in its subfolder
        #draw_roi_on_image(image1, beam1, beam_folder, f'{os.path.splitext(file1)[0]}_Beam{beam_index + 1}')

    # Create a CSV file in the subfolder to store the intensity differences for each Beam
    csv_file_path_beam = os.path.join(beam_folder, f'EIT_Intensity_Beam{beam_index + 1}.csv')
    with open(csv_file_path_beam, 'w', newline='') as csv_file_beam:
        csv_writer_beam = csv.writer(csv_file_beam)
        csv_writer_beam.writerow(['File', 'EIT_Intensity'])

        # Write the image number and intensity difference to the CSV file for each Beam
        for file_name, intensity_diff_beam in zip(file_names_beam, intensity_differences_beam):
            csv_writer_beam.writerow([file_name, intensity_diff_beam])

    # Plot a graph with normalized data and fitted curve for each Beam
    plt.plot(file_names_beam, intensity_differences_beam, marker='o', label=f'Intensity curve - Beam{beam_index + 1}')
    plt.xlabel('Power in mW')
    #plt.xticks([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000])
    #plt.ylabel('Intensity')
    #plt.title(f'Intensity vs Power - Beam{beam_index + 1}')
    plt.grid(True)
    plt.legend()

    # Save the plot to the subfolder for each Beam
    plot_file_path_beam = os.path.join(beam_folder, f'Intensity vs Power_Beam{beam_index + 1}.TIFF')
    plt.savefig(plot_file_path_beam)

    # Show the plot for each Beam
    plt.show()

    # Plot the graph with EIT and Background values for each Beam
    plt.figure()  # Create a new figure for each Beam
    plt.plot(file_names_beam, EIT_values_beam, marker='x', label=f'EIT Beam {beam_index + 1}')
    plt.plot(file_names_beam, background_values_beam, marker='x', label=f'Background Beam {beam_index + 1}')
    plt.xlabel('Power in mW')
    #plt.xticks([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000])
    plt.ylabel('Intensity')
    plt.title(f'Background & Intensity vs Power - Beam{beam_index + 1}')
    plt.grid(True)
    plt.legend()

    # Save the plot for each Beam in its corresponding folder
    plot_file_path_beam = os.path.join(main_output_folder, f'Beam{beam_index + 1}', f'Background & Intensity vs Power_Beam{beam_index + 1}.png')
    os.makedirs(os.path.dirname(plot_file_path_beam), exist_ok=True)
    plt.savefig(plot_file_path_beam)
    plt.show()

# Plot a graph with normalized data and fitted curve for all Beams together
plt.figure(figsize=(12, 8))

# Adjust the layout to add more space between subplots
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)

# Process each Beam separately
for beam_index, (beam1, beam2) in enumerate(zip(roi_coords, roi_coords)):
    # Create a subfolder for each Beam inside the main output folder
    beam_folder = os.path.join(main_output_folder, f'Beam{beam_index + 1}')
    os.makedirs(beam_folder, exist_ok=True)

    # Initialize lists to store file names and intensity differences for each Beam
    file_names_beam = []
    intensity_differences_beam = []

    # Initialize lists to store EIT and Background values for each Beam
    EIT_values_beam = []
    background_values_beam = []

    # Process the image pairs
    for file1, file2 in zip(natsort.natsorted(image_files_1), natsort.natsorted(image_files_2)):
        # Extract the image numbers from the filenames
        image_num1 = float(os.path.splitext(file1)[0])

        # Construct the file paths for the image pairs
        image_path1 = os.path.join(image_dir_1, file1)
        image_path2 = os.path.join(image_dir_2, file2)

        # Load the U16 images
        image1 = cv2.imread(image_path1, cv2.IMREAD_UNCHANGED)
        image2 = cv2.imread(image_path2, cv2.IMREAD_UNCHANGED)

        # Calculate the intensity difference for each Beam and accumulate the results
        intensity_diff = calculate_intensity_difference(image1, image2, beam1, beam2)

        # Calculate EIT and Background values
        EIT_value = calculate_EIT(image1, beam1)
        background_value = calculate_background(image2, beam2)
        EIT_values_beam.append(EIT_value)
        background_values_beam.append(background_value)

        # Append the image number and intensity difference to the lists for each Beam
        file_names_beam.append(image_num1)
        intensity_differences_beam.append(intensity_diff)

        # Draw and save the image with the Beam for each Beam in its subfolder
        #draw_rois_on_image(image1, beam1, beam_folder, f'{os.path.splitext(file1)[0]}_Beam{beam_index + 1}')

    # Create a CSV file in the subfolder to store the intensity differences for each Beam
    csv_file_path_beam = os.path.join(beam_folder, f'EIT_Intensity_Beam{beam_index + 1}.csv')
    with open(csv_file_path_beam, 'w', newline='') as csv_file_beam:
        csv_writer_beam = csv.writer(csv_file_beam)
        csv_writer_beam.writerow(['File', 'EIT_Intensity'])

        # Write the image number and intensity difference to the CSV file for each Beam
        for file_name, intensity_diff_beam in zip(file_names_beam, intensity_differences_beam):
            csv_writer_beam.writerow([file_name, intensity_diff_beam])

    # Create a subplot for each Beam
    plt.subplot(3, 2, beam_index + 1)
    plt.plot(file_names_beam, intensity_differences_beam, marker='o', label=f'Beam{beam_index + 1}')
    #plt.xlabel('Power in mW')
    #plt.ylabel('Intensity')
    plt.title(f'Beam{beam_index + 1}')
    plt.grid(True)
    #plt.legend()

# Save the overall plot to a file
overall_plot_file_path = os.path.join(main_output_folder, 'Overall_Intensity_vs_Power.png')
plt.savefig(overall_plot_file_path)

# Show the overall plot
plt.show()
