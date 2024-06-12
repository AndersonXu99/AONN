import numpy as np

class ImageProcessor:
    def __init__(self, Image, Row, Column, Parameter, ind_begin, lengthpre, W_theory, partnum, Rescale, C_x, C_y):
        self.Image = Image.T
        self.Row = Row
        self.Column = Column
        self.I = self.Row * self.Column
        self.Parameter = Parameter
        self.ind_begin = ind_begin
        self.lengthpre = lengthpre # show I field in the LABView user interface
        self.W_theory = W_theory
        self.partnum = partnum
        self.Rescale = Rescale
        self.C_x = C_x
        self.C_y = C_y

    def measuresequence(self):
        xwidth = int(self.Parameter[2])
        ywidth = int(self.Parameter[3])
        xbegin = int(self.Parameter[0])
        ybegin = int(self.Parameter[1])

        Image_pre = np.zeros((self.lengthpre*xwidth, ywidth))
        index = 0
        
        for ii in range(self.Column):
            for i in range(self.Row):
                ind = (i)*self.Column+ii
                if ind >= self.ind_begin and ind < self.ind_begin+self.lengthpre:
                    index += 1
                    x_s = xbegin + round( (ii) * (self.Parameter[4] / (self.Column-1) ) + (i) * self.Parameter[6] / (self.Row-1))
                    y_s = ybegin + round( (i) *(self.Parameter[5]/(self.Row-1))+(ii)*self.Parameter[7]/(self.Column-1))
                    part = self.Image[x_s : x_s + xwidth, y_s : y_s+ywidth]
                    Image_pre[(index-1) * xwidth : index * xwidth, :] = part
    
        return Image_pre.T

    def measurecal(self):
        xwidth = int(self.Parameter[2])
        ywidth = int(self.Parameter[3])
        xbegin = int(self.Parameter[0])
        ybegin = int(self.Parameter[1])
        Image_pre = np.zeros((self.Row*self.Column*xwidth, ywidth))

        weight_measured = np.zeros(self.Row*self.Column)

        for ii in range(self.Column):
            for i in range(self.Row):
                ind = (i) * self.Column+ii
                if ind > self.Row*self.Column:
                    break

                x_s = xbegin + round((ii) * (self.Parameter[4] / (self.Column-1)) + (i) * self.Parameter[6] / (self.Row-1))
                y_s = ybegin + round((i)*(self.Parameter[5]/(self.Row-1))+(ii)*self.Parameter[7]/(self.Column-1))

                part = self.Image[x_s : x_s+xwidth, y_s : y_s + ywidth]
                Image_pre[(ind-1)*xwidth:ind*xwidth, :] = part
                weight_measured[ind] = np.sum(part)
        return Image_pre.T, weight_measured

    def disablepoint(disable, RandomSet, Row, Column, Center_x, Center_y):
        ind_dis = []
        if disable != 0:
            for x in range(Column):
                for y in range(Row):
                    if ((x-Center_x))**2+(y-Center_y)**2 < (disable**2):
                        ind_dis.append((y)*Column+x)
        Set = RandomSet.copy()
        Set[:, ind_dis] = 0
        return ind_dis, Set

    def main(self, ifshow):
        if (self.partnum-1) % 15 > 5:
            self.Rescale *= ((self.partnum-1) % 15 + 35) / 35

        if ifshow == 1:
            Image_pre = self.measuresequence()
            weight_measured = []
            Rescale_c = np.sum(self.Image) / (self.Column*self.Row) * 0.6
        else:
            disable = 2
            ind_dis, _ = self.disablepoint(disable)
            _, weight_measured = self.measurecal()
            weight_measured[ind_dis] = 0
            Image_pre = [0, 0, 0, 0]
            Rescale_c = weight_measured / (self.Row * self.Column - len(ind_dis))

        if len(weight_measured) != 0:
            self.W_theory = self.W_theory[:len(self.Row*self.Column)]
            weight_measured_nor = weight_measured / self.Rescale
            weight_error = self.W_theory - weight_measured_nor
            std_error = np.std(weight_error)
        else:
            std_error = 1
            weight_error = 1

        return Image_pre, weight_measured, Rescale_c, weight_error, std_error