import numpy as np
import matplotlib.pyplot as plt
from GS_algorithm_first_iteration import gsw_output
from GS_algorithm2 import *
from SLM_Control import *
from showSLMPreview import showSLMPreview
from dcam_live_capturing import *
from beam_locator import *
import time

### ---------------------------------------------------------------------------------------------------------------- ###
### AONN - GSW Main Interface                                                                                        ###
### Author: Anderson Xu                                                                                              ###
### ---------------------------------------------------------------------------------------------------------------- ###

# Original MATLAB code
# Input: Dim, SLM_length, SLM_width, Row, Column, balance, time, interval, weight, Pattern_last, 
# Correction, part, part_width, part_length

# cd 'C:\Linear Operations\code\linear_iteration_1stSLM'

# size_real=[part_width, part_length];
# Pattern=zeros(SLM_length,SLM_width);

# %Pattern=mod([1:SLM_width],5)/5*2*pi;
# %Pattern=repmat(Pattern,SLM_length,1);

# temp=zeros(1,Column*Row);
# temp(1:length(weight))=weight;
# weight_shaped=reshape(temp,Column,Row);

# weight_shaped=flipud(weight_shaped);

# if time==0 
#   [Pattern_part,phi]=gsw_output(size_real, weight_shaped,interval);
# else
#  balance=0;
#   [Pattern_part,phi]=gs_iteration_modified(size_real,weight_shaped,interval,Pattern_last,balance);
# end


# x=Dim(1)-1-mod(part-1,Dim(1));
# y=floor((part-1)/Dim(1));

# Pattern(y*size_real(2)+1:(y+1)*size_real(2),x*size_real(1)+1:(x+1)*size_real(1))=Pattern_part;

# if size(Pattern)~=[1080,1920]
# Pattern=Pattern';
# end

# Pattern=mod(Pattern+Correction,2*pi);
# %Pattern=Correction;

### ---------------------------------------------------------------------------------------------------------------- ###
### Initialization                                                                                                   ###
### ---------------------------------------------------------------------------------------------------------------- ###
number_of_rows = 2
number_of_columns = 2
# weight = np.ones(number_of_columns * number_of_rows) / (number_of_columns * number_of_rows)
weight = np.array([0.25, 0.25, 0.25, 0.25])
w0 = 1
interval = 50
error_allowed = 0.01 
error_allowed_file_writing = 0.1 
Dim = np.array([12, 12])
# size_real = np.array([1920, 1080]) 
# using the HOLOEYE GAEA-2, the resolution is 3840 x 2160
size_real = np.array([3840, 2160])
Overall_Pattern = np.zeros((2160, 3840), dtype=np.float64)
error = 0

# initiate the camera and SLM instances
slm = SLMControler()
dcam_capture = DcamLiveCapturing(iDevice = 0)

### ---------------------------------------------------------------------------------------------------------------- ###
### Function Definitions                                                                                             ###
### ---------------------------------------------------------------------------------------------------------------- ###
def pattern_placement(part, Pattern, Pattern_part):
    x = int(Dim[0] - 1 - np.mod(part - 1, Dim[0]))
    y = int(np.floor((part - 1) / Dim[0]))  

    start_x = x * size_real[0]
    end_x = (x+1) * size_real[0]
    start_y = y * size_real[1]
    end_y = (y+1) * size_real[1]
 
    Pattern[start_y : end_y, start_x : end_x] = Pattern_part
    return Pattern



### ---------------------------------------------------------------------------------------------------------------- ###
### Main Function                                                                                                    ###
### ---------------------------------------------------------------------------------------------------------------- ###
size_real = size_real / Dim 
size_real = size_real.astype(int)
temp = np.zeros(number_of_columns * number_of_rows)
temp[:len(weight)] = weight  
weight_shaped = np.reshape(temp, (number_of_columns, number_of_rows))
weight_shaped = np.flipud(weight_shaped)

# capture the background image before phase patterns are displayed 
background_image = dcam_capture.capture_single_frame()

# get the initial beam locations by running the GSW algorithm without feedback and running the beam locator function
initial_Pattern = np.zeros((2160, 3840), dtype=np.float64)
for part in range(1, 145):
    # since we are not improving, we are only iterating over this 2 times, first time to use the gsw_output function, and second time to use the gs_iteration_modified function
    for time in range (0, 2):
        balance = 0
        if time == 0:
            [Pattern_part, phi] = gsw_output(size_real, weight_shaped, interval, number_of_rows, number_of_columns, w0)
            Pattern_last = phi
        else:
            [Pattern_part, phi] = gs_iteration_modified(size_real, weight_shaped, interval, Pattern_last, balance, w0)
            Pattern_last = phi

    print("Part Number: " + str(part))
    
    x = int(Dim[0] - 1 - np.mod(part - 1, Dim[0]))
    y = int(np.floor((part - 1) / Dim[0]))  

    start_x = x * size_real[0]
    end_x = (x+1) * size_real[0]
    start_y = y * size_real[1]
    end_y = (y+1) * size_real[1]
 
    initial_Pattern[start_y : end_y, start_x : end_x] = Pattern_part

# divide each value in Pattern by 2 * pi
initial_Pattern = np.mod(initial_Pattern, 2 * np.pi)
slm.display_data(initial_Pattern)

### Beam Locator ###
# show the live stream for validation and to capture the image used for cursor locating
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

    locator = beam_locator(captured_image, number_of_rows, number_of_columns)

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

    # print("Beam Corners:", locator.beam_corners)
else:
    print("No image captured.")
####################


# stores the beam locations in a local variables to use
beam_locations = locator.beam_locations
beam_corners = locator.beam_corners

for part in range(1, 145):

    # for each part, we are only concerned with that specific part and the rest of the phase pattern would be zero
    # so we first initialize a phase pattern with all zeros
    Pattern = np.zeros((2160, 3840), dtype=np.float64)

    # maximum of 80 iterations
    for iteration in range (0, 81):

        # check if the std is lower than the given set number
        if error < error_allowed: 
            break 

        if iteration == 0:
            [Pattern_part, phi] = gsw_output(size_real, weight_shaped, interval, number_of_rows, number_of_columns, w0)
            Pattern_last = phi

            # placing the pattern at the correct location on the whole SLM phase pattern
            Pattern = pattern_placement(part, Pattern, Pattern_part)   
            Pattern = np.mod(Pattern, 2 * np.pi)

            # now we want to display the pattern on the SLM and capture an image of it
            slm.display_data(Pattern)

            # time delay of 0.5 seconds 
            time.sleep(0.5)

            # capture an image of the SLM
            captured_image = dcam_capture.capture_single_frame()
            
            # subtract the background image from the captured image
            captured_image = captured_image - background_image

            # calculate the intensity of each beam using the locations found and stored in a file
            # 

        else:
            [Pattern_part, phi] = gs_iteration_modified(size_real, weight_shaped, interval, Pattern_last, error, w0)
            Pattern_last = phi

            Pattern = pattern_placement(part, Pattern, Pattern_part)


    # after the iterations, we will place the most optimal pattern part in the overall_pattern

