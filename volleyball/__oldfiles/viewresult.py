import cv2
import numpy as np
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as anm
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import csv
import calc_viewresult as calview

class ViewResult:
    Line = [[0, 1], [0, 2], [1, 2], [7, 8], [8, 10], [9, 10], [7, 9], [7, 11], [8, 18], [9, 12], [10, 19], [11, 12],
            [12, 19], [18, 19], [18, 11], [11, 13],
            [12, 14], [13, 14], [13, 15], [14, 16], [15, 16], [15, 17], [16, 17], [18, 20], [19, 21], [20, 21],
            [20, 23], [21, 24], [23, 24], [23, 25], [24, 25],
            [3, 5], [3, 6], [5, 6]]

    def __init__(self, inputdata, referencedata):#input[joint][time][dim]
        self.__input = inputdata
        self.__reference = referencedata
        self.__input_time = inputdata.shape[1]
        self.__reference_time = referencedata.shape[1]
        self.__dimension = inputdata.shape[2]

        if self.__dimension != referencedata.shape[2]:
            print "invalid combination between input and reference: dimension error"
            sys.exit()

        self.__correspondent_points = {}


    def __view_update(self, time, data, title, extX, extY, extZ):
        if time != 0:
            self.__ax.cla()

        self.__ax.set_xlim(extX)
        self.__ax.set_ylim(extY)
        self.__ax.set_zlim(extZ)

        for line in self.Line:
            self.__ax.plot([data[line[0]][time][0], data[line[1]][time][0]],
                           [data[line[0]][time][1], data[line[1]][time][1]],
                           [data[line[0]][time][2], data[line[1]][time][2]], "-", color='black')
        self.__ax.scatter3D(data[:, time, 0], data[:, time, 1], data[:, time, 2], ".", color='black')
        plt.title(title)

    def view3d_reference(self, filepath=''):
        xmax = np.nanmax(self.__reference[:, :, 0])
        xmin = np.nanmin(self.__reference[:, :, 0])
        ymax = np.nanmax(self.__reference[:, :, 1])
        ymin = np.nanmin(self.__reference[:, :, 1])
        zmax = np.nanmax(self.__reference[:, :, 2])
        zmin = np.nanmin(self.__reference[:, :, 2])
        aspectFT = np.array([xmax - xmin, ymax - ymin, zmax - zmin])
        add_tmp = aspectFT.max()
        addx = add_tmp - (xmax - xmin)
        addy = add_tmp - (ymax - ymin)
        addz = add_tmp - (zmax - zmin)

        ani = anm.FuncAnimation(self.__fig, self.__view_update, fargs=(self.__reference, 'reference', [xmin, xmax + addx], [ymin, ymax + addy], [zmin, zmax + addz]),
                                interval=20, frames=self.__reference_time - 1)
        if filepath != '':
            ani.save(filepath)
        plt.show()
        plt.cla()
        self.__ax.cla()

    def view3d_input(self, filepath=''):
        xmax = np.nanmax(self.__input[:, :, 0])
        xmin = np.nanmin(self.__input[:, :, 0])
        ymax = np.nanmax(self.__input[:, :, 1])
        ymin = np.nanmin(self.__input[:, :, 1])
        zmax = np.nanmax(self.__input[:, :, 2])
        zmin = np.nanmin(self.__input[:, :, 2])
        aspectFT = np.array([xmax - xmin, ymax - ymin, zmax - zmin])
        add_tmp = aspectFT.max()
        addx = add_tmp - (xmax - xmin)
        addy = add_tmp - (ymax - ymin)
        addz = add_tmp - (zmax - zmin)

        ani = anm.FuncAnimation(self.__fig, self.__view_update, fargs=(
        self.__input, 'input', [xmin, xmax + addx], [ymin, ymax + addy], [zmin, zmax + addz]),
                                interval=20, frames=self.__input_time - 1)
        if filepath != '':
            ani.save(filepath)
        plt.show()
        plt.cla()
        self.__ax.cla()

    def set_result(self, joint_name, input_joint, reference_joint, correspondent_point, constraint):
        self.__correspondent_points[joint_name] = {"inp_j_num":input_joint, "ref_j_num":reference_joint, "corr_point": correspondent_point}
        self.__constraint = constraint

    def __calc_DPresult(self, calculate_slope, forRef):
        rgblist = {}

        input_time = None
        if not forRef:
            input_time = self.__input_time

        for key, correspondent_point in self.__correspondent_points.iteritems():
            joint_num = correspondent_point["ref_j_num"]

            if self.__constraint in ["asym", "default"]:
                rgblist[joint_num] = hsv_to_rgb(calview.calc_hsv_asym(correspondent_point, calculate_slope, input_time))

            else:
                print "not defined to calculate this constraint = {}".format(self.__constraint)
                exit()

        return rgblist


    def view_DPResult(self, forRef=True, calculate_slope=True, filepath=''):
        joints = len(self.__correspondent_points)
        if joints == 0:
            print "DP results were not setted, reference .set_result()"
            sys.exit()

        if self.__dimension == 2:
            self.__fig = plt.figure()
            plt.xlabel('x')
            plt.ylabel('y')

        elif self.__dimension == 3:
            self.__fig = plt.figure()
            self.__ax = Axes3D(self.__fig)
            # axis label
            self.__ax.set_xlabel('x')
            self.__ax.set_ylabel('y')
            self.__ax.set_zlabel('z')

        else:
            print "the dimension of data must be 2 or 3"
            sys.exit()


        """
        const int size = 101;
        // sigma of gauss function
        const double sigma = 20.0;
        // matrix to record hsv
        var hsvMap = new Mat(new Size(size, size), MatType.CV_8UC3);
        for (int i = -50; i <= 50; ++i)
        {
            for (int j = -50; j <= 50; ++j)
            {
                // value of gauss function
                double g = Math.Exp(-(Math.Pow(i, 2) + Math.Pow(j, 2))/(2.0*Math.Pow(sigma, 2)));
                // calculate huered is 0, blue is 120 in OpenCV
                byte hue = (byte)(120*(1.0 - g));
                hsvMap.Set(i+50, j+50, new Vec3b(hue, 255, 255));
            }
        }
        // convert hsv to bgr
        var bgrMap = new Mat(new Size(size, size), MatType.CV_8UC3);
        Cv2.CvtColor(hsvMap, bgrMap, ColorConversionCodes.HSV2BGR);
        
        float[,] heatmap = player.batteddistribution(tabControl1.SelectedIndex == 0, comboBox_kind.SelectedIndex);
                //var hsvmap = OpenCvSharp.Extensions.BitmapConverter.ToMat(canvas);
                using (hsvmap = new OpenCvSharp.Mat(new OpenCvSharp.Size(400, 300), OpenCvSharp.MatType.CV_8UC3))
                using (bgrMap = new OpenCvSharp.Mat(new OpenCvSharp.Size(400, 300), OpenCvSharp.MatType.CV_8UC3))
                {
                    for (int x = 0; x < 400; x++)
                    {
                        for (int y = 50; y < 250; y++)
                        {
                            byte hue = (byte)(120 * (1.0 - heatmap[x, y - 50]));
                            hsvmap.Set(y, x, new OpenCvSharp.Vec3b(hue, 255, 255));
                        }
                    }
                    OpenCvSharp.Cv2.CvtColor(hsvmap, bgrMap, OpenCvSharp.ColorConversionCodes.HSV2BGR);
                    OpenCvSharp.Cv2.Ellipse(bgrMap, new OpenCvSharp.Point(200, 250), new OpenCvSharp.Size(200, 200), 0, -45, -135, new OpenCvSharp.Scalar(0, 0, 0), 1);
                    OpenCvSharp.Cv2.Ellipse(bgrMap, new OpenCvSharp.Point(200, 250), new OpenCvSharp.Size(100, 100), 0, -45, -135, new OpenCvSharp.Scalar(0, 0, 0), 1);
                    OpenCvSharp.Cv2.Line(bgrMap, new OpenCvSharp.Point(200, 250), new OpenCvSharp.Point(341, 109), new OpenCvSharp.Scalar(0, 0, 0));
                    OpenCvSharp.Cv2.Line(bgrMap, new OpenCvSharp.Point(200, 250), new OpenCvSharp.Point(59, 109), new OpenCvSharp.Scalar(0, 0, 0));
                    canvas = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(bgrMap);
                }
        """
        rgblist = self.__calc_DPresult(calculate_slope, forRef)

        #view
        xmax = np.nanmax(self.__reference[:, :, 0])
        xmin = np.nanmin(self.__reference[:, :, 0])
        ymax = np.nanmax(self.__reference[:, :, 1])
        ymin = np.nanmin(self.__reference[:, :, 1])
        zmax = np.nanmax(self.__reference[:, :, 2])
        zmin = np.nanmin(self.__reference[:, :, 2])
        aspectFT = np.array([xmax - xmin, ymax - ymin, zmax - zmin])
        add_tmp = aspectFT.max()
        addx = add_tmp - (xmax - xmin)
        addy = add_tmp - (ymax - ymin)
        addz = add_tmp - (zmax - zmin)

        ani = anm.FuncAnimation(self.__fig, self.__DPview_update, fargs=(
            self.__reference, 'reference', [xmin, xmax + addx], [ymin, ymax + addy], [zmin, zmax + addz], rgblist),
                                interval=20, frames=self.__reference_time - 1)
        #if filepath != '':
        #    ani.save(filepath)
        plt.show()
        plt.cla()
        self.__ax.cla()

    def __DPview_update(self, time, data, title, extX, extY, extZ, rgblist):
        if time != 0:
            self.__ax.cla()

        self.__ax.set_xlim(extX)
        self.__ax.set_ylim(extY)
        self.__ax.set_zlim(extZ)

        for line in self.Line:
            self.__ax.plot([data[line[0]][time][0], data[line[1]][time][0]],
                           [data[line[0]][time][1], data[line[1]][time][1]],
                           [data[line[0]][time][2], data[line[1]][time][2]], "-", color='black')

        scatters = [self.__ax.scatter3D(data[j, time, 0], data[j, time, 1], data[j, time, 2], ".", color='black') for j in range(len(data))]
        for joint, rgb in rgblist.iteritems():
            scatters[joint].remove()
            self.__ax.scatter3D(data[joint, time, 0], data[joint, time, 1], data[joint, time, 2], ".", color=rgb[time])

        plt.title(title + ' frame=' + str(time))

    def view_DPResult_colorbar(self, calculate_slope=True, filepath='', show=True, forRef=True):
        joints = len(self.__correspondent_points)
        if joints == 0:
            print "DP results were not setted, reference .set_result()"
            sys.exit()

        self.__fig = plt.figure()
        if forRef:
            plt.xlabel('reference')

            rgblist = self.__calc_DPresult(calculate_slope, forRef)

            for joint, rgb in rgblist.iteritems():
                J = joint
                for time in range(1, self.__reference_time):
                    plt.barh([joint], [1], left=[time], color=rgb[time])
            red = plt.barh([J], [1], color='red', label="fast")
            blu = plt.barh([J], [1], color='blue', label="slow")
            bla = plt.barh([J], [1], color='black', label="correspond")
            plt.legend([red, blu, bla], ["fast", "slow", "correspond"], loc='upper right')
        else:
            plt.xlabel('input')

            rgblist = self.__calc_DPresult(calculate_slope, forRef)

            for joint, rgb in rgblist.iteritems():
                #red = plt.barh([joint], [1], color='red', label="fast")
                #blu = plt.barh([joint], [1], color='blue', label="slow")
                #bla = plt.barh([joint], [1], color='black', label="correspond")
                #gre = plt.barh([joint], [1], color='white', label="no reference")
                J = joint
                for time in range(1, self.__input_time):
                    plt.barh([joint], [1], left=[time], color=rgb[time])
            red = plt.barh([J], [1], color='red', label="fast")
            blu = plt.barh([J], [1], color='blue', label="slow")
            bla = plt.barh([J], [1], color='black', label="correspond")
            gre = plt.barh([J], [1], color=[0, 1, 0], label="no reference")
            plt.legend([red, blu, bla, gre], ["fast", "slow", "correspond", "no reference"], loc='upper right')

        if show:
            plt.show()
        if filepath != "":
            plt.savefig(filepath)

    def draw_allgraph(self, notall=False, savefile=""):
        joints = len(self.__correspondent_points)
        if joints == 0:
            print "DP results were not setted, reference .set_result()"
            sys.exit()

        self.__fig = plt.figure()

        # plt.plot([0, self.__input_time], [0, self.__input_time], 'black', linestyle='dashed')
        for joint_name in self.__correspondent_points.keys():
            if notall:
                plt.cla()
            plt.xlabel('reference')
            plt.ylabel('input')
            plt.xlim([0, self.__input_time])
            plt.ylim([0, self.__input_time])
            plt.gca().set_aspect('equal', adjustable='box')
            plt.vlines([self.__reference_time], 0, self.__input_time, linestyles='dashed')
            tmp_x = np.linspace(0, self.__reference_time, self.__reference_time + 1)

            correspondent_point = self.__correspondent_points[joint_name]["corr_point"]
            plt.plot([0, self.__reference_time],
                     [correspondent_point[0][1],
                      self.__reference_time + correspondent_point[0][1]], 'black',
                     linestyle='dashed')
            # x = np.array([self.correspondent_point[i][0] for i in range(len(self.correspondent_point))], dtype=np.int)
            # y = np.array([self.correspondent_point[i][1] for i in range(len(self.correspondent_point))], dtype=np.int)
            x = [correspondent_point[i][0] for i in range(len(correspondent_point))]
            y = [correspondent_point[i][1] for i in range(len(correspondent_point))]
            plt.plot(x, y, label=joint_name)
            plt.legend()
            if savefile == "":
                plt.show()
            else:
                if notall:
                    plt.savefig(savefile + joint_name + ".png")
                else:
                    plt.savefig(savefile + "all.png")




class ViewOnly:
    Line = [[0, 1], [0, 2], [1, 2], [7, 8], [8, 10], [9, 10], [7, 9], [7, 11], [8, 18], [9, 12], [10, 19], [11, 12],
            [12, 19], [18, 19], [18, 11], [11, 13],
            [12, 14], [13, 14], [13, 15], [14, 16], [15, 16], [15, 17], [16, 17], [18, 20], [19, 21], [20, 21],
            [20, 23], [21, 24], [23, 24], [23, 25], [24, 25],
            [3, 5], [3, 6], [5, 6]]

    def __init__(self, datapath, dim, remove_rows=None, remove_cols=None):
        input_data = []

        if dim == 2:
            self.__fig = plt.figure()
            plt.xlabel('x')
            plt.ylabel('y')
        elif dim == 3:
            self.__fig = plt.figure()
            self.__ax = Axes3D(self.__fig)
            # axis label
            self.__ax.set_xlabel('x')
            self.__ax.set_ylabel('y')
            self.__ax.set_zlabel('z')
        else:
            print "the dimension of data must be 2 or 3"
            sys.exit()

        with open(datapath, "rb") as f:
            reader = csv.reader(f)
            for row in reader:
                if remove_cols is not None:
                    del row[remove_cols]
                if not row:
                    continue
                # input_data.append(row)

                tmp = []
                for data in row:
                    if not data:
                        tmp.append('nan')
                        continue
                    tmp.append(data)
                input_data.append(tmp)


        if remove_rows is not None:
            del input_data[remove_rows]

        input_data = np.array(input_data, dtype=np.float)

        return_inp_data = []

        for data_index in range(0, input_data.shape[1], dim):
            tmp_row = []
            for time in range(0, input_data.shape[0]):
                tmp = []
                for i in range(dim):
                    tmp.append(input_data[time][i + data_index])
                tmp_row.append(tmp)
            return_inp_data.append(tmp_row)

        self.__data = np.array(return_inp_data)
        self.__time = self.__data.shape[1]

    def __update(self, time, data, title, extX, extY, extZ):
        if time != 0:
            self.__ax.cla()

        self.__ax.set_xlim(extX)
        self.__ax.set_ylim(extY)
        self.__ax.set_zlim(extZ)

        for line in self.Line:
            self.__ax.plot([data[line[0]][time][0], data[line[1]][time][0]],
                           [data[line[0]][time][1], data[line[1]][time][1]],
                           [data[line[0]][time][2], data[line[1]][time][2]], "-", color='black')
        self.__ax.scatter3D(data[:, time, 0], data[:, time, 1], data[:, time, 2], ".", color='black')
        plt.title(title)

    def view(self, savepath='', saveonly=False):
        xmax = np.nanmax(self.__data[:, :, 0])
        xmin = np.nanmin(self.__data[:, :, 0])
        ymax = np.nanmax(self.__data[:, :, 1])
        ymin = np.nanmin(self.__data[:, :, 1])
        zmax = np.nanmax(self.__data[:, :, 2])
        zmin = np.nanmin(self.__data[:, :, 2])
        aspectFT = np.array([xmax - xmin, ymax - ymin, zmax - zmin])
        add_tmp = aspectFT.max()
        addx = add_tmp - (xmax - xmin)
        addy = add_tmp - (ymax - ymin)
        addz = add_tmp - (zmax - zmin)

        ani = anm.FuncAnimation(self.__fig, self.__update, fargs=(
            self.__data, 'data', [xmin, xmax + addx], [ymin, ymax + addy], [zmin, zmax + addz]),
                                interval=20, frames=self.__time - 1)
        if savepath != '':
            ani.save(savepath)
            if saveonly:
                plt.cla()
                self.__ax.cla()
                return
        plt.show()
        plt.cla()
        self.__ax.cla()
