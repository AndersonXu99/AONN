import numpy as np
import matplotlib.pyplot as plt
from GS_algorithm_first_iteration import gsw_output
from GS_algorithm2 import *
from SLM_Control import *
from showSLMPreview import showSLMPreview
from dcam_live_capturing import *
from beam_locator import *
import time

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

### Initilization ###
number_of_rows = 2
number_of_columns = 2
# weight = np.ones(number_of_columns * number_of_rows) / (number_of_columns * number_of_rows)
weight = np.array([0.25, 0.25, 0.25, 0.25])
w0 = 1
interval = 50

Dim = np.array([12, 12])
# the weight matrix will be a 1D array of 25 elements with uniform weights that sum to 1

# size_real = np.array([1920, 1080]) 
# using the HOLOEYE GAEA-2, the resolution is 3840 x 2160
size_real = np.array([3840, 2160])

size_real = size_real / Dim 
size_real = size_real.astype(int)
temp = np.zeros(number_of_columns * number_of_rows)
temp[:len(weight)] = weight  
weight_shaped = np.reshape(temp, (number_of_columns, number_of_rows))
weight_shaped = np.flipud(weight_shaped)


Dim = np.array([12, 12])
size_real = np.array([3840, 2160]) 
size_real = size_real / Dim 

# make sure size_real is an integer
size_real = size_real.astype(int)
Pattern = np.zeros((2160, 3840), dtype=np.float64)

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

    
    x = int(Dim[0] - 1 - np.mod(part - 1, Dim[0]))
    y = int(np.floor((part - 1) / Dim[0]))  

    start_x = x * size_real[0]
    end_x = (x+1) * size_real[0]
    start_y = y * size_real[1]
    end_y = (y+1) * size_real[1]
 
    Pattern[start_y : end_y, start_x : end_x] = Pattern_part

# divide each value in Pattern by 2 * pi
Pattern = np.mod(Pattern, 2 * np.pi)

# initiate the SLM controler using the SLMControler class
slm = SLMControler()
slm.display_data(Pattern)

# apply a 0.5 second delay
time.sleep(0.5)

# after the phase pattern is displayed, capture a frame of the from the camera
# Create DcamLiveCapturing instance
dcam_capture = DcamLiveCapturing(iDevice = 0)
background_image = dcam_capture.capture_single_frame()

# Capture live images
dcam_capture = DcamLiveCapturing(iDevice = 0)
captured_image = dcam_capture.capture_live_images()

slm.close()

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
