import cv2
import numpy as np
from dcam_live_capturing import *

class CrosshairLocator:
    """
    A class used to locate and draw crosshairs on an image.

    ...

    Attributes
    ----------
    image : ndarray
        The image on which the crosshairs are drawn.
    cursor_locations : list
        A list of tuples representing the (x, y) coordinates of the crosshairs.
    dragging : bool
        A flag indicating whether a crosshair is currently being dragged.
    current_cursor : int
        The index of the current cursor being dragged.
    crosshair_length : int
        The length of the crosshair lines.
    rows : int
        The number of rows in the image.
    cols : int
        The number of columns in the image.

    Methods
    -------
    draw_crosshair(img, center):
        Draws a crosshair at the specified center location on the image.

    display_image_with_crosshairs():
        Displays the image with crosshairs at the cursor locations.
    """
    def __init__(self, image, number_of_rows, number_of_cols):
        self.image = image
        self.cursor_locations = [(100, 100), (1700, 100), (200, 200), (100, 1700)]  # Initialize cursor locations
        self.dragging = False
        self.current_cursor = None
        self.crosshair_length = 40  # Set the length of the crosshair lines
        self.rows = number_of_rows
        self.cols = number_of_cols

    def draw_crosshair(self, img, center):
        # Draw vertical line
        cv2.line(img, (center[0], center[1] - self.crosshair_length // 2),
                 (center[0], center[1] + self.crosshair_length // 2), (0, 255, 0), 2)
        # Draw horizontal line
        cv2.line(img, (center[0] - self.crosshair_length // 2, center[1]),
                 (center[0] + self.crosshair_length // 2, center[1]), (0, 255, 0), 2)

    def display_image_with_crosshairs(self):
        cv2.namedWindow('Image with Crosshairs', cv2.WINDOW_NORMAL)  # Resizable window

        while True:
            img = self.image.copy()

            # Draw crosshairs at cursor locations
            for loc in self.cursor_locations:
                self.draw_crosshair(img, loc)

            cv2.imshow('Image with Crosshairs', img)

            # Set mouse callback function
            cv2.setMouseCallback('Image with Crosshairs', self.mouse_callback)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if cursor is clicked
            for idx, loc in enumerate(self.cursor_locations):
                if abs(loc[0] - x) < self.crosshair_length // 2 and abs(loc[1] - y) < self.crosshair_length // 2:
                    self.dragging = True
                    self.current_cursor = idx
                    break
        elif event == cv2.EVENT_MOUSEMOVE:
            # Update cursor location if dragging
            if self.dragging:
                self.cursor_locations[self.current_cursor] = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            # Stop dragging
            self.dragging = False
            self.current_cursor = None

    def get_cursor_locations(self):

        # sort the cursor locations in the following order, top left, top right, bottom left, bottom right
        self.cursor_locations = sorted(self.cursor_locations, key=lambda x: x[0])
        self.cursor_locations = sorted(self.cursor_locations, key=lambda x: x[1])

        return self.cursor_locations
    
    def calculate_all_beam_locations(self):
        # from the first and third elements of the cursor_locations list, we can calculate the diameter of a beam
        # the first element of the array is the top left of the beam and the third element represents the bottom right corner of the beam
        self.beam_diameter = np.sqrt((self.cursor_locations[2][0] - self.cursor_locations[0][0])**2 + (self.cursor_locations[2][1] - self.cursor_locations[0][1])**2)
        print("Beam Diameter:", self.beam_diameter)

        # from the first and second element of the list, these are the top left corners of the top left most and top right most beams
        # find the horizontal intervals of all beams
        self.horizontal_intervals = (self.cursor_locations[1][0] - self.cursor_locations[0][0]) / (self.cols - 1)
        
        # now to find the vertical intervals
        self.vertical_intervals = (self.cursor_locations[3][1] - self.cursor_locations[0][1]) / (self.rows - 1)

        # now from these horizontal and vertical intervals, we can calculate the top left corners of all 25 beams and create a box using the diameter calculated as well
        # store the corners in a 25 x 4 array, for each beam, store the four corners of the box

        # Initialize the array with zeros
        self.total_num_beams = self.rows * self.cols
        self.beam_corners = np.zeros((self.total_num_beams, 2, 2), dtype=float)

        # Iterate over the rows
        for i in range(self.rows):
            # Iterate over the columns
            for j in range(self.cols):
                # Calculate the top left corner of the current box
                top_left = [self.cursor_locations[0][0] + j * self.horizontal_intervals, self.cursor_locations[0][1] + i * self.vertical_intervals]
                
                # Calculate the bottom right corner
                bottom_right = [top_left[0] + self.beam_diameter, top_left[1] + self.beam_diameter]
                
                # Store the corners in the array
                self.beam_corners[i * self.rows + j] = [top_left, bottom_right]


# The main function that will call the class the display the image with crosshairs after the user has selected the cursor locations, the beam locations will be calculated 
# and stored in a text file
def main():
    number_of_rows = 5
    number_of_columns = 5

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

        locator = CrosshairLocator(captured_image, number_of_rows, number_of_columns)

        # Display image with crosshairs and allow user interaction
        locator.display_image_with_crosshairs()

        # Get cursor locations
        cursor_locations = locator.get_cursor_locations()

        # save the cursor locations to a text file
        with open("cursor_locations.txt", "w") as f:
            for loc in cursor_locations:
                f.write(f"{loc[0]}, {loc[1]}\n")

        # print("Cursor Locations:", cursor_locations)

        locator.calculate_all_beam_locations()

        print("Beam Corners:", locator.beam_corners)
    else:
        print("No image captured.")