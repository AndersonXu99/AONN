# #--------------------------------------------------------------------#
# #                                                                    #
# # Copyright (C) 2020 HOLOEYE Photonics AG. All rights reserved.      #
# # Contact: https://holoeye.com/contact/                              #
# #                                                                    #
# # This file is part of HOLOEYE SLM Display SDK.                      #
# #                                                                    #
# # You may use this file under the terms and conditions of the        #
# # "HOLOEYE SLM Display SDK Standard License v1.0" license agreement. #
# #                                                                    #
# # This code modified by Anderson Xu                                  #
# #                                                                    #
# #--------------------------------------------------------------------#

import detect_heds_module_path
from holoeye import slmdisplaysdk
from showSLMPreview import showSLMPreview
import random

class SLMControler: 
    """
    A class used to control the Spatial Light Modulator (SLM).

    Attributes
    ----------
    slm : SLMInstance
        An instance of the SLM.
    dataWidth : int
        The width of the SLM in pixels.
    dataHeight : int
        The height of the SLM in pixels.

    Methods
    -------
    __init__():
        Initializes the SLM controller, opens the SLM window, checks the library version,
        and opens the SLM preview window in "Fit" mode. It also reserves memory for the data
        that will be displayed on the SLM.
    display_data(data):
        Takes a 2D array `data` as input, checks if its shape matches the SLM's shape, and
        displays the data on the SLM.
    close():
        Waits until the SLM process is closed, then unloads the SDK.
    """

    def __init__(self):
        # Open the SLM window:
        self.slm = slmdisplaysdk.SLMInstance()
        # Check if the library implements the required version
        if not self.slm.requiresVersion(3):
            exit(1)

        error = self.slm.open()
        assert error == slmdisplaysdk.ErrorCode.NoError, self.slm.errorString(error)

        # Open the SLM preview window in "Fit" mode:
        # Please adapt the file showSLMPreview.py if preview window
        # is not at the right position or even not visible.

        showSLMPreview(self.slm, scale=0.0)

        # Calculate e.g. a vertical blazed grating:
        # this was used to create a blazed grating
        # self.blazePeriod = 77

        # Reserve memory for the data:
        self.dataWidth  = self.slm.width_px
        self.dataHeight = self.slm.height_px

    def display_data (self, data):
        # check if the data has the correct shape
        assert data.shape == (self.dataHeight, self.dataWidth), "Data shape does not match SLM shape"
        # Displaying the data on the SLM using the .showData() method
        error = self.slm.showPhasevalues(data)
        assert error == slmdisplaysdk.ErrorCode.NoError, self.slm.errorString(error)

# If your IDE terminates the python interpreter process after the script is finished, the SLM content
# will be lost as soon as the script finishes.

    def close(self):
        # Wait until the SLM process is closed:
        print("Waiting for SDK process to close. Please close the tray icon to continue ...")
        error = self.slm.utilsWaitUntilClosed()
        assert error == slmdisplaysdk.ErrorCode.NoError, self.slm.errorString(error)

        # Unloading the SDK may or may not be required depending on your IDE:
        self.slm = None


# Use Case
# slm = SLMControler()

# data = slmdisplaysdk.createFieldSingle(slm.dataWidth, slm.dataHeight)
# # print("dataWidth = " + str(dataWidth))
# # print("dataHeight = " + str(dataHeight))

# # creating a random phase pattern
# for y in range(slm.dataHeight):
#     row = data[y]

#     for x in range(slm.dataWidth):
#         row[x] = random.random() 

# slm.display_data(data)
# slm.close()


# # Import the SLM Display SDK:
# import detect_heds_module_path
# from holoeye import slmdisplaysdk

# # Initializes the SLM library
# slm = slmdisplaysdk.SLMInstance()

# # Check if the library implements the required version
# if not slm.requiresVersion(3):
#     exit(1)

# # Detect SLMs and open a window on the selected SLM
# error = slm.open()
# assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

# # Open the SLM preview window in non-scaled mode:
# # Please adapt the file showSLMPreview.py if preview window
# # is not at the right position or even not visible.
# from showSLMPreview import showSLMPreview
# showSLMPreview(slm, scale=1.0)

# # Configure blazed grating:
# period = 40
# shift = 0
# phaseScale = 1.0
# phaseOffset = 0.0
# vertical = True

# # Show blazed grating on SLM:
# if vertical:
#     error = slm.showGratingVerticalBlaze(period, shift, phaseScale, phaseOffset)
# else:
#     error = slm.showGratingHorizontalBlaze(period, shift, phaseScale, phaseOffset)

# assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

# # If your IDE terminates the python interpreter process after the script is finished, the SLM content
# # will be lost as soon as the script finishes.

# # You may insert further code here.

# # Wait until the SLM process is closed:
# print("Waiting for SDK process to close. Please close the tray icon to continue ...")
# error = slm.utilsWaitUntilClosed()
# assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

# slm.close()
# # Unloading the SDK may or may not be required depending on your IDE:
# slm = None
