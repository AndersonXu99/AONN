import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import time

def gsw_output(size_real, weight, interval, Row, Column):
    """
    Function to generate the output of the Gerchberg-Saxton (GS) algorithm.

    Parameters
    ----------
    size_real : list
        The real size of the SLM.
    weight : numpy.ndarray
        The weight matrix for the GS algorithm.
    interval : int
        The interval for the GS algorithm.

    Returns
    -------
    Image_SLM : numpy.ndarray
        The image to be displayed on the SLM.
    phi : numpy.ndarray
        The phase matrix from the GS algorithm.
    """
    def IntensityMeasure(B, position):
        rc = (position[:, 1] - position[:, 0] + 1) / interval
        power = np.zeros((rc[0], rc[1]))
        for i in range(rc[0]):
            for ii in range(rc[1]):
                x = position[0, 0] + interval / 2 + (i - 1) * interval
                y = position[1, 0] + interval / 2 + (ii - 1) * interval
                ratio_x = np.ceil(ratio * size_real[0] / size_real[1])
                power[i, ii] = np.sum(np.abs(B[int(x - ratio_x / 2):int(x + ratio_x / 2 - 1), int(y - ratio / 2):int(y + ratio / 2 - 1)]) ** 2)
        power_sum = np.sum(np.abs(B) ** 2)
        return power, power_sum

    def GS_algorithm(phase, g, position):
        size_ = (size_part - 1) / 2
        X, Y = np.meshgrid(np.arange(-size_[0], size_[0] + 1), np.arange(-size_[1], size_[1] + 1))
        A0 = np.exp(-((X.T) ** 2) / (1000 ** 2) - (Y.T ** 2) / (1000 ** 2)) * np.exp(1j * phase)
        B0 = fftshift(fft2(A0, (size_part[0], size_part[1])))
        A0 = A0[int(real_rect[0, 0]):int(real_rect[0, 1]), int(real_rect[1, 0]):int(real_rect[1, 1])]
        B = fftshift(fft2(A0, (size_part[0], size_part[1])))
        ak = np.sqrt(IntensityMeasure(B, position)[0])
        g_next = (np.sqrt(weight) / np.sum(np.sqrt(weight))) / (ak / np.sum(ak)) * g
        weight_next = g_next * np.sqrt(weight)
        at, _ = Multibeam(weight_next / np.mean(weight_next) * 0.9)
        D = (at) * np.exp(1j * np.angle(B0))
        E = ifft2(ifftshift(D))
        Output = np.angle(E)
        return Output, g_next

    def Multibeam(weight):
        row, column = weight.shape
        # interval is the distance between two beams also the diameter of a single beam
        # the single_r will be the radius of a single beam
        single_r = (interval - 1) / 2
        # creating a mesh grid with the center being (0, 0) and 
        single_x, single_y = np.meshgrid(np.arange(-single_r, single_r + 1), np.arange(-single_r, single_r + 1))
        # creating a Gaussian beam that centers around the center of the meshgrid
        singlepattern = np.exp(-2 * (single_x ** 2 + single_y ** 2) / w0 ** 2)
        # repeating this Gaussian beam arranged in a grid that has row number of rows and column number of columns
        Multi = np.tile(singlepattern, (row, column))

        for i in range(row): # i = 0, 1, 2, 3, ..., row - 1
            for ii in range(column):
                # multiplying each beam with the given weight, so each output beam will have the corresponding weight 
                # fix the indexing when translating from MATLAB
                Multi[(i) * interval: (i+1) * interval, (ii) * interval: (ii + 1) * interval] = Multi[(i) * interval: (i + 1) * interval, (ii) * interval: (ii + 1) * interval] * weight[i, ii] # weight is a matrix with row x col
        
        # Multi_x = 
        Multi_x, Multi_y = Multi.shape
        Multipattern = np.zeros(size_part)

        # gives the poisition ranges of the phase pattern
        position = np.array([[np.floor(size_part[0] / 2) - np.floor(Multi_x / 2), np.floor(size_part[0] / 2) + np.floor(Multi_x / 2)], [np.floor(size_part[1] / 2) - np.floor(Multi_y / 2), np.floor(size_part[1] / 2) + np.floor(Multi_y / 2)]])
        Multipattern[int(position[0, 0]):int(position[0, 1]), int(position[1, 0]):int(position[1, 1])] = Multi
        return Multipattern, position

    # start_time = time.time()

    w0 = 1
    if size_real[0] > 500:
        ratio = 2
    else:
        ratio = 4

        
    # size_part = [1, 1] * size_real[0] * ratio

    weight_shaped= np.reshape(weight[ : Row*Column], (Column, Row))         # not part of gsw
    weight_shaped= np.transpose(weight_shaped)                              # not part of gsw
    weight_shaped= np.flipud(weight_shaped)                                 # not part of gsw

    size_part = [1, 1] * int(round(size_real[0] * ratio))
    padnum = (size_part - size_real) / 2
    real_rect = np.array([[padnum[0] + 1, padnum[0] + size_real[0]], [padnum[1] + 1, padnum[1] + size_real[1]]])
    _, position = Multibeam(np.sqrt(weight))
    Phase0 = np.random.rand(size_part[0], size_part[1])
    g = np.ones(weight.shape)
    phi, _ = GS_algorithm(Phase0, g, position)
    for nn in range(10):
        phi, g = GS_algorithm(phi, g, position)
    Phase_f = phi[int(real_rect[0, 0]):int(real_rect[0, 1]), int(real_rect[1, 0]):int(real_rect[1, 1])]
    Phase_n = np.mod(Phase_f, 2 * np.pi)
    Image_SLM = Phase_n.T
    # print("Execution time: ", time.time() - start_time)
    return Image_SLM, phi