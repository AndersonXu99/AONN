#--------------------------------------------------------------------#
#                                                                    #
# Copyright (C) 2020 HOLOEYE Photonics AG. All rights reserved.      #
# Contact: https://holoeye.com/contact/                              #
#                                                                    #
# This file is part of HOLOEYE SLM Display SDK.                      #
#                                                                    #
# You may use this file under the terms and conditions of the        #
# "HOLOEYE SLM Display SDK Standard License v1.0" license agreement. #
#                                                                    #
#--------------------------------------------------------------------#

import detect_heds_module_path
from holoeye import slmdisplaysdk
from showSLMPreview import showSLMPreview
import random

# Open the SLM window:
slm = slmdisplaysdk.SLMInstance()
# Check if the library implements the required version
if not slm.requiresVersion(3):
    exit(1)

error = slm.open()
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

# Open the SLM preview window in "Fit" mode:
# Please adapt the file showSLMPreview.py if preview window
# is not at the right position or even not visible.

showSLMPreview(slm, scale=0.0)

# Calculate e.g. a vertical blazed grating:
# this was used to create a blazed grating
# blazePeriod = 77

# Reserve memory for the data:
dataWidth  = slm.width_px
dataHeight = slm.height_px
data = slmdisplaysdk.createFieldSingle(dataWidth, dataHeight)
# print("dataWidth = " + str(dataWidth))
# print("dataHeight = " + str(dataHeight))

# creating a random phase pattern
for y in range(dataHeight):
    row = data[y]

    for x in range(dataWidth):
        row[x] = random.random() 

# Displaying the data on the SLM using the .showData() method
error = slm.showData(data)
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

# If your IDE terminates the python interpreter process after the script is finished, the SLM content
# will be lost as soon as the script finishes.

# You may insert further code here.

# Wait until the SLM process is closed:
print("Waiting for SDK process to close. Please close the tray icon to continue ...")
error = slm.utilsWaitUntilClosed()
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

# Unloading the SDK may or may not be required depending on your IDE:
slm = None