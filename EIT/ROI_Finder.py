import csv

import cv2
import numpy as np

# Global variables to store circle parameters
circles = []

# Function to calculate distance between two points
def distance(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

# Function to calculate circle parameters from two points
def circle_from_two_points(p1, p2):
    center_x = (p1[0] + p2[0]) // 2
    center_y = (p1[1] + p2[1]) // 2
    radius = int(distance(p1, p2) / 2)
    return (center_x, center_y), radius

# Callback function for mouse events
def draw_circle(event, x, y, flags, param):
    global circles

    if event == cv2.EVENT_LBUTTONDOWN:
        circles.append({'center': None, 'radius': 0, 'second_point': None, 'first_point': (x, y)})

    elif event == cv2.EVENT_LBUTTONUP:
        if circles:
            circles[-1]['second_point'] = (x, y)
            circles[-1]['center'], circles[-1]['radius'] = circle_from_two_points(circles[-1]['first_point'], (x, y))

# Undo function
def undo_last_circle():
    global circles
    if circles:
        circles.pop()

# Load image
image_path = r"C:\Users\zxq220007\Box\Quantum Optics Lab\TeTON OANN Testbed\Data 2024\Apr 16 2024\5X5 Trans5 50mm cell 120C 290MHz 3037MHz\ROI.tif"
image = cv2.imread(image_path)
if image is None:
    print("Error: Image not found.")
    exit()

# Create a window and bind the mouse callback function
cv2.namedWindow('image', cv2.WINDOW_NORMAL)  # Resizable window
cv2.setMouseCallback('image', draw_circle)

# Main loop for displaying the image and handling user input
while True:
    # Make a copy of the original image to draw on
    image_copy = image.copy()

    # Draw circles
    for circle in circles:
        if circle['second_point']:
            cv2.circle(image_copy, circle['center'], circle['radius'], (0, 255, 0), 2)

    # Display the image with the circles
    cv2.imshow('image', image_copy)

    # Check for key events
    key = cv2.waitKey(1) & 0xFF
    # Press 'q' to quit
    if key == ord('q'):
        break
    # Press 'u' to undo
    elif key == ord('u'):
        undo_last_circle()


# Write circle parameters to a text file
output_file_path = "circle_parameters.txt"


def write_circle_information_to_csv(circles, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['Center_X', 'Center_Y', 'Radius']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i, circle in enumerate(circles, start=1):
            writer.writerow({
                'Center_X': circle['center'][0],
                'Center_Y': circle['center'][1],
                'Radius': circle['radius']
            })

output_csv_file = "circle_info.csv"  # Output CSV file path
write_circle_information_to_csv(circles, output_csv_file)
# Close all OpenCV windows
cv2.destroyAllWindows()
