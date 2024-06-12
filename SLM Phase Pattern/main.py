import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# importing supporting files from the same directory
from dcam_live_capturing import *
from beam_locator import *
import holoeye.slmdisplaysdk as slm
from GS_algorithm_first_iteration import *
from GS_algorithm2 import *
from SLM_Control import *

### Initilization ###
number_of_rows = 5
number_of_columns = 5
Dim = np.array([12, 12])
# the weight matrix will be a 1D array of 25 elements with uniform weights that sum to 1
weight = np.ones(number_of_columns * number_of_rows) / (number_of_columns * number_of_rows)
interval = 50
size_real = np.array([1920, 1080]) 
size_real = size_real / Dim 
temp = np.zeros(number_of_columns * number_of_rows)
temp[:len(weight)] = weight  
weight_shaped = np.reshape(temp, (number_of_columns, number_of_rows))
weight_shaped = np.flipud(weight_shaped)

# Before any phase patterns are initialized, we want to first capture the background image
# Create DcamLiveCapturing instance
dcam_capture = DcamLiveCapturing(iDevice = 0)
background_image = dcam_capture.capture_single_frame()

# Capture live images
dcam_capture = DcamLiveCapturing(iDevice = 0)
captured_image = dcam_capture.capture_live_images()

# iterate through the 144 parts of the SLM

# part goes from 1 to 144

for part in range(1, 145):
    # for each part, we want to generate the pattern using the two GS algorithms for at most 80 times
    for time in range(1, 81):
        if time == 1:
            # Run the first iteration of the GS algorithm to show the location of the beams
            Pattern_part, phi = gsw_output(size_real, weight_shaped, interval, number_of_rows, number_of_columns)
            Pattern_last = Pattern_part

            if part == 1 and time == 1:
                # on the very first iteration, we want to run the beam locator to get the cursor locations
                beam_locator = CrosshairLocator(captured_image, number_of_rows, number_of_columns)   
                beam_locator.run()

        else:
            Pattern_part, phi = gs_iteration_modified(size_real, weight_shaped, interval, Pattern_last, e)

        # after we have obtain the pattern, we want to display it on the SLM using the SLM_Control class
        slm = SLMControler()
        
        # check if the pattern is in the correct shape
        if Pattern_part.shape != (slm.dataWidth, slm.dataHeight):
            # print out an error messgae
            print("Pattern shape is not correct")
            break

        # Pattern_last = Pattern_part
        slm.display_data(Pattern_part)
        # slm.close()

        # put a delay of 0.5 seconds to make sure the SLM is properly displaying
        time.sleep(0.5) 
        

# after this, we want to take a measurement of the beam using the camera, then calculate the measured weight matrix
def measured_weight ():
    pass

    

# size_real = np.array([1920, 1080]) // Dim
# Pattern = np.zeros((1080, 1920))
# Pattern_last = np.zeros((1080, 1920))
# e = 0

# temp = np.zeros(number_of_columns * number_of_rows)
# temp[:len(weight)] = weight  # W1
# weight_shaped = np.reshape(temp, (number_of_columns, number_of_rows))

# weight_shaped = np.flipud(weight_shaped)

# # if time == 0:
# #     Pattern_part, phi = gsw_output(size_real, weight_shaped, interval)
# #     Pattern_last = Pattern_part
# # else:
# #     Pattern_part, phi = gs_iteration_modified(size_real, weight_shaped, interval, Pattern_last, e)  # after the first iteration, we have a measured weight
# #     Pattern_last = Pattern_part

# x = Dim[0] - 1 - np.mod(part - 1, Dim[0])
# y = np.floor((part - 1) / Dim[0])

# Pattern[int(y*size_real[1]):int((y+1)*size_real[1]), int(x*size_real[0]):int((x+1)*size_real[0])] = Pattern_part

# if Pattern.shape != (1080, 1920):
#     Pattern = Pattern.T

# Pattern = np.mod(Pattern + Correction, 2 * np.pi)

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

# cd 'C:\Linear Operations\code\linear_iteration_1stSLM'

# size_real=[part_width, part_length];
# Pattern=zeros(SLM_length,SLM_width);

# Pattern=mod([1:SLM_width],5)/5*2*pi;
# Pattern=repmat(Pattern,SLM_length,1);


# temp=zeros(1,Column*Row);
# temp(1:length(weight))=weight;
# weight_shaped=reshape(temp,Column,Row);


# if time==0 
#   [Pattern_part,phi]=gsw_output(size_real, weight_shaped,interval);
# else
#  balance=0;
#   [Pattern_part,phi]=gs_iteration_modified(size_real,weight_shaped,interval,Pattern_last,balance);
# end

# grating = -mod([1:part_width], 40)*2*pi/40;
# grating = repmat(grating, part_length,1);


# %areaxy=[1720, 880];
# areaxy=[1, 1];


# Pattern(areaxy(2):areaxy(2)+part_length-1,areaxy(1):areaxy(1)+part_width-1)=Pattern_part+grating;

# %Pattern(1:2000,1:3600)=repmat(Pattern_part+grating,5,9);


# if size(Pattern)~=[SLM_length,SLM_width]
# Pattern=Pattern';
# end
