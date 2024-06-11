import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# importing supporting files from the same directory
from dcam_live_capturing import *
from beam_locator import *
import holoeye.slmdisplaysdk as slm
from GS_algorithm1 import *
from GS_algorithm2 import *

### Initilization ###
number_of_rows = 5
number_of_columns = 5
Dim = np.array([12, 12])
weight = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
interval = 50

# Before any phase patterns are initialized, we want to first capture the background image
# Create DcamLiveCapturing instance
dcam_capture = DcamLiveCapturing(iDevice = 0)
background_image = dcam_capture.capture_single_frame()

# after the background image is captured, we want to display a uniform distribution on the SLM display

def show_phase_pattern_on_slm(phase_array):
    """
    Displays a phase pattern on the HOLOEYE SLM using official methods.

    Args:
        phase_array (numpy.ndarray): 1D array containing the phase values.

    Returns:
        None
    """
    try:
        # Initialize SLM display (replace with actual initialization)
        slm_device = slm.SLMDevice()

        # Reshape the 1D phase array to match the SLM resolution (replace with actual resolution)
        slm_width, slm_height = slm_device.get_resolution()
        phase_matrix = np.reshape(phase_array, (slm_height, slm_width))

        # Set the phase pattern on the SLM (replace with actual method to set phase)
        slm_device.set_phase(phase_matrix)

        print("Phase pattern displayed on HOLOEYE SLM successfully.")
    except ImportError:
        print("Error: holoeye.slmdisplaysdk library not installed.")
    except Exception as e:
        print(f"Error: {e}")


# Capture live images
dcam_capture = DcamLiveCapturing(iDevice = 0)
captured_image = dcam_capture.capture_live_images()



# Check if an image was captured
if captured_image is not None:
    print("Image captured successfully.")
    print(captured_image)

    # Display the captured image using OpenCV
    cv2.imshow("Captured Image", captured_image)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()  # Close the window
else:
    print("No image captured.")

# now that the image has been captured, we want to display the image so the user can use it indicate the location of the beams
# Create CrosshairLocator instance
locator = CrosshairLocator(captured_image, number_of_rows, number_of_columns)

# Display image with crosshairs and allow user interaction
locator.display_image_with_crosshairs()

# Get cursor locations
cursor_locations = locator.get_cursor_locations()
# print("Cursor Locations:", cursor_locations)

locator.calculate_all_beam_locations()

print("Beam Corners:", locator.beam_corners)

def generate_pattern_time_zero(size_real, weight_shaped, interval):
    """
    Generates the pattern for time = 0.

    Args:
        size_real (numpy.ndarray): Array containing the real size of the pattern.
        weight_shaped (numpy.ndarray): Array containing the shaped weights.
        interval (int): Interval value.

    Returns:
        numpy.ndarray: Generated pattern.
    """
    Pattern_part, phi = gsw_output(size_real, weight_shaped, interval)
    Pattern_last = Pattern_part
    return Pattern_part, Pattern_last, phi


def generate_pattern_other_time(size_real, weight_shaped, interval, Pattern_last, e):
    """
    Generates the pattern for time other than 0.

    Args:
        size_real (numpy.ndarray): Array containing the real size of the pattern.
        weight_shaped (numpy.ndarray): Array containing the shaped weights.
        interval (int): Interval value.
        Pattern_last (numpy.ndarray): Last pattern generated.
        e (int): Error value.

    Returns:
        numpy.ndarray: Generated pattern.
    """
    Pattern_part, phi = gs_iteration_modified(size_real, weight_shaped, interval, Pattern_last, e)
    # Pattern_last = Pattern_part
    return Pattern_part, phi

# after this, we want to take a measurement of the beam using the camera, then calculate the measured weight matrix
def measured_weight ():
    pass

def process_weights(W_mea, W_theory, Rescale, partnum, Times):
    # Times is the number of frames that are captured
    if partnum % 12 > 5:
        Rescale *= 1.5

    temp = W_mea.shape

    if temp[0] != 1 and temp[1] != 1:
        W_theory = W_theory[:W_mea.shape[1]]
        weight_measure = np.mean(W_mea)
        weight_measure /= Rescale
        errorstd = np.std(weight_measure - W_theory)
        errormean = np.mean(weight_measure - W_theory)
    else:
        W_theory = W_theory[:W_mea.shape[1]*Times]
        W_theory = np.reshape(W_theory, (Times, W_mea.shape[1]))
        weight_measure = W_mea / Rescale
        errorstd = np.std(weight_measure - W_theory)
        errormean = np.mean(weight_measure - W_theory)

    return W_theory, weight_measure, errorstd, errormean
    

size_real = np.array([1920, 1080]) // Dim
Pattern = np.zeros((1080, 1920))
Pattern_last = np.zeros((1080, 1920))
e = 0

temp = np.zeros(number_of_columns * number_of_rows)
temp[:len(weight)] = weight  # W1
weight_shaped = np.reshape(temp, (number_of_columns, number_of_rows))

weight_shaped = np.flipud(weight_shaped)

# if time == 0:
#     Pattern_part, phi = gsw_output(size_real, weight_shaped, interval)
#     Pattern_last = Pattern_part
# else:
#     Pattern_part, phi = gs_iteration_modified(size_real, weight_shaped, interval, Pattern_last, e)  # after the first iteration, we have a measured weight
#     Pattern_last = Pattern_part

x = Dim[0] - 1 - np.mod(part - 1, Dim[0])
y = np.floor((part - 1) / Dim[0])

Pattern[int(y*size_real[1]):int((y+1)*size_real[1]), int(x*size_real[0]):int((x+1)*size_real[0])] = Pattern_part

if Pattern.shape != (1080, 1920):
    Pattern = Pattern.T

Pattern = np.mod(Pattern + Correction, 2 * np.pi)

######

# cd 'C:\Linear Operations\code\linear_iteration_1stSLM'

# l=Row*Column;

# Image=Image';

# if mod(partnum-1,15)>5

# Rescale=Rescale*(mod(partnum-1,15)+35)/35;

# end



# if ifshow==1


# Image_pre=measuresequence(Image,Row,Column,Parameter,ind_begin,lengthpre);
# weight_measured=[ ];
# Rescale_c=sum(Image(:))/(Column*Row)*0.6;



# else

# disable=2
# [ind_dis,~]=disablepoint(disable,W_theory,Row,Column,C_x,C_y);
# [~,weight_measured]=measurecal(Image,Row,Column,Parameter,l);
# weight_measured(ind_dis)=0;
# Image_pre=[0 0 0 0];
# Rescale_c=weight_measured/(Row*Column-length(ind_dis));



# end

# if isempty(weight_measured)==0

# W_theory=W_theory(1:length(Row*Column));
# weight_measured_nor=weight_measured./Rescale;
# weight_error=W_theory-weight_measured_nor;

# std_error=std(weight_error);

# else
# std_error=1;
# weight_error=1;
# end

######

# clearvars -EXCEPT W_mea W_theory error Rescale partnum Times

# if mod(partnum-1,12)>5

# Rescale=Rescale*1.5;

# else

# Rescale = Rescale
# end


# ;

# temp=size(W_mea);

# if temp(2)~=1 && temp(1)~=1
# W_theory=W_theory(1:size(W_mea,2));
# weight_measure=mean(W_mea);
# weight_measure=weight_measure./Rescale;
# errorstd=std(weight_measure-W_theory);
# errormean=mean(weight_measure-W_theory);

# else
# W_theory=W_theory(1:size(W_mea,2)*Times);
# W_theory = reshape(W_theory, Times, size(W_mea,2));
# weight_measure=W_mea./Rescale;
# errorstd=std(weight_measure-W_theory);
# errormean=mean(weight_measure-W_theory);

# end