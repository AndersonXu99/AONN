import cv2
import numpy as np
from multiprocessing import Pool

# Function to calculate sum of intensities for a grid cell
def calculate_grid_sum(args):
    image, i, j, num_cols, grid_height, grid_width, threshold = args
    # Define the coordinates of the current grid cell
    top_left_y = i * grid_height
    top_left_x = j * grid_width
    bottom_right_y = (i + 1) * grid_height
    bottom_right_x = (j + 1) * grid_width
    
    # Extract the current grid cell from the image
    grid_cell = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    
    # Apply threshold to filter out bright spots
    grid_cell_filtered = np.where(grid_cell > threshold, grid_cell, 0)
    
    # Calculate the sum of intensities for the filtered grid cell
    grid_sum = np.sum(grid_cell_filtered)
    
    return grid_sum

# Load the image
image = cv2.imread('your_image.jpg', cv2.IMREAD_GRAYSCALE)

# Define the number of rows and columns for the grid
num_rows = 2
num_cols = 7

# Get the height and width of the image
height, width = image.shape

# Calculate the size of each grid cell
grid_height = height // num_rows
grid_width = width // num_cols

# Define the threshold to filter out bright spots
threshold = 200    

# Initialize a list to store arguments for each grid cell
grid_args = [(image, i, j, num_cols, grid_height, grid_width, threshold) for i in range(num_rows) for j in range(num_cols)]

# Use multiprocessing Pool to calculate sums concurrently
with Pool() as p:
    grid_sums = p.map(calculate_grid_sum, grid_args)

# Print the sums for each grid cell
print("Grid Sums:")
for i, grid_sum in enumerate(grid_sums):
    print(f"Grid {i+1}: {grid_sum}")
