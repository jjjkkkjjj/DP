import numpy as np
import csv
from scipy.interpolate import CubicSpline as cs
from scipy.interpolate import interp1d
import os

# Data.joints[joint] = [time, dim]
class Data:
    def __init__(self, interpolate='linear'):
        """
                @param joints : has values including a missing value, dictionary
                                joints[joint name][time,dim]
                @param trcpath : trc file path

                @param frame_max : maximum frame max in the video

                @param interpolate : the method of interpolation [linear, spline, None]
        """
        self.name = ""
        self.joints = None
        self.trcpath = None
        self.dir = None
        self.frame_max = None
        self.interpolate = interpolate
        self.lines = None
        pass

    def set_from_trc(self, path, lines='volleyball'):
        with open(path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            next(reader)
            next(reader)
            forthline = next(reader)
            joints = forthline[2::3]
            if joints[-1] == '':
                joints = joints[:-1]

            data = np.genfromtxt(f, delimiter='\t', skip_header=6, missing_values=' ')
            self.joints = {}

            for index, joint in enumerate(joints):
                if self.interpolate is not None:
                    self.joints[joint] = self.__interpolate(data, index)
                else:
                    self.joints[joint] = data[:, 3*index + 2:3*index + 5]

            self.frame_max = self.joints[joints[0]].shape[0]

            self.trcpath = path
            self.dir = os.path.dirname(self.trcpath)
            self.name = self.trcpath.split('/')[-1]

            if lines == 'volleyball':
                """
                self.Points = ["head", "R_ear", "L_ear", "sternum", "C7", "R_rib", "L_rib", "R_ASIS", "L_ASIS",
                               "R_PSIS",
                               "L_PSIS",
                               "R_frontshoulder", "R_backshoulder", "R_in_elbow", "R_out_elbow", "R_in_wrist",
                               "R_out_wrist",
                               "R_hand",
                               "L_frontshoulder", "L_backshoulder", "L_in_elbow", "L_out_elbow", "D_UA?", "L_in_wrist",
                               "L_out_wrist",
                               "L_hand"]
                self.lines = [[0, 1], [0, 2], [1, 2], [7, 8], [8, 10], [9, 10], [7, 9], [7, 11], [8, 18], [9, 12],
                              [10, 19], [11, 12],
                              [12, 19], [18, 19], [18, 11], [11, 13],
                              [12, 14], [13, 14], [13, 15], [14, 16], [15, 16], [15, 17], [16, 17], [18, 20], [19, 21],
                              [20, 21],
                              [20, 23], [21, 24], [23, 24], [23, 25], [24, 25],
                              [3, 5], [3, 6], [5, 6]]
                """
                Lines = [['head', 'R_ear'], ['head', 'L_ear'], ['R_ear', 'L_ear'],
                              ['R_ASIS', 'L_ASIS'], ['L_ASIS', 'L_PSIS'], ['R_PSIS', 'L_PSIS'], ['R_ASIS', 'R_PSIS'],
                              ['R_ASIS', 'R_frontshoulder'], ['L_ASIS', 'L_frontshoulder'], ['R_PSIS', 'R_backshoulder'], ['L_PSIS', 'L_backshoulder'],
                              ['R_frontshoulder', 'R_backshoulder'], ['R_backshoulder', 'L_backshoulder'],
                              ['L_frontshoulder', 'L_backshoulder'], ['L_frontshoulder', 'R_frontshoulder'],
                              ['R_frontshoulder', 'R_in_elbow'], ['R_backshoulder', 'R_out_elbow'],
                              ['R_in_elbow', 'R_out_elbow'], ['R_in_elbow', 'R_in_wrist'], ['R_out_elbow', 'R_out_wrist'], ['R_in_wrist', 'R_out_wrist'],
                              ['R_in_wrist', 'R_hand'], ['R_out_wrist', 'R_hand'],
                              ['L_frontshoulder', 'L_in_elbow'], ['L_backshoulder', 'L_out_elbow'], ['L_in_elbow', 'L_out_elbow'],
                              ['L_in_elbow', 'L_in_wrist'], ['L_out_elbow', 'L_out_wrist'], ['L_in_wrist', 'L_out_wrist'], ['L_in_wrist', 'L_hand'],
                              ['L_out_wrist', 'L_hand'], ['sternum', 'R_rib'], ['sternum', 'L_rib'], ['R_rib', 'L_rib']]

                jointNames = list(self.joints.keys())
                self.lines = []
                for line in Lines:
                    self.lines.append([jointNames.index(line[0]), jointNames.index(line[1])])

            else:
                print("Warning: {0} is unknown lines".format(lines))

    def check(self):
        pass

    def __interpolate(self, data, index):
        if self.interpolate == 'linear':
            x, y, z = data[:, 3 * index + 2], data[:, 3 * index + 3], data[:, 3 * index + 4]
            time = np.where(~np.isnan(x))[0]
            if time.size == 0:
                return data[:, 3 * index + 2:3 * index + 5]

            interp1d_x = interp1d(time, x[time], fill_value='extrapolate')
            interp1d_y = interp1d(time, y[time], fill_value='extrapolate')
            interp1d_z = interp1d(time, z[time], fill_value='extrapolate')

            time = [i for i in range(x.shape[0])]

            return np.vstack((interp1d_x(time), interp1d_y(time), interp1d_z(time))).T
            # print(np.sum(np.isnan(interp1d_x(time))), np.sum(np.isnan(interp1d_y(time))), np.sum(np.isnan(interp1d_z(time))))
            # if np.where(np.isnan(self.joints[joint]))[0].size > 0:
            #    print(joint, self.trcpath)

        elif self.interpolate == 'spline':
            x, y, z = data[:, 3 * index + 2], data[:, 3 * index + 3], data[:, 3 * index + 4]
            time = np.where(~np.isnan(x))[0]
            if time.size == 0:
                return data[:, 3 * index + 2:3 * index + 5]

            spline_x = cs(time, x[time])
            spline_y = cs(time, y[time])
            spline_z = cs(time, z[time])

            time = [i for i in range(x.shape[0])]

            return np.vstack((spline_x(time), spline_y(time), spline_z(time))).T
        else:
            raise ValueError("{0} is not defined as interpolation method".format(self.interpolate))

    def setvalues(self, dataname, x, y, z, jointNames, jointaxis=0, dir=None, lines='baseball'):
        if type(x).__name__ != 'ndarray' or type(y).__name__ != 'ndarray' or type(z).__name__ != 'ndarray':
            raise TypeError("x, y, z must be ndarray")

        if type(jointNames).__name__ != 'list':
            raise TypeError("joitNames must be list")

        if x.shape != y.shape or y.shape != z.shape or z.shape != y.shape:
            raise ValueError("x, y, z must have same shape")

        if x.shape[jointaxis] != len(jointNames):
            raise ValueError("data shape[{0}] and jointNames[size:{1}] must be same".format(x.shape[jointaxis], len(jointNames)))


        self.name = dataname
        if dir is not None:
            self.dir = dir

        data = []

        if jointaxis == 0:
            data.append(np.zeros(x.shape[1]))
            data.append(np.zeros(x.shape[1]))
            for jointindex in range(len(jointNames)):
                data.append(x[jointindex, :])
                data.append(y[jointindex, :])
                data.append(z[jointindex, :])
        elif jointaxis == 1:
            data.append(np.zeros(x.shape[0]))
            data.append(np.zeros(x.shape[0]))
            for jointindex in range(len(jointNames)):
                data.append(x[:, jointindex])
                data.append(y[:, jointindex])
                data.append(z[:, jointindex])
        else:
            raise IndexError("jointaxis must be 0 or 1")
        data = np.array(data).T

        self.joints = {}

        for index, joint in enumerate(jointNames):
            if self.interpolate is not None:
                self.joints[joint] = self.__interpolate(data, index)
            else:
                self.joints[joint] = data[:, 3 * index + 2:3 * index + 5]

        self.frame_max = self.joints[jointNames[0]].shape[0]

        if lines == 'baseball':
            lines = [['LTOE', 'LANK'], ['LTIB', 'LANK'], ['LASI', 'LPSI'],  # around ankle
                     ['RTOE', 'RANK'], ['RTIB', 'RANK'], ['RASI', 'RPSI'],  # "
                     ['LASI', 'RASI'], ['LPSI', 'RPSI'], ['LHEE', 'LANK'], ['RHEE', 'RANK'], ['LHEE', 'LTOE'],
                     ['RHEE', 'RTOE'],  # around hip
                     ['LHEE', 'LTIB'], ['RHEE', 'RTIB'],  # connect ankle to knee
                     ['LKNE', 'LTIB'], ['LKNE', 'LTHI'], ['LASI', 'LTHI'], ['LPSI', 'LTHI'],  # connect knee to hip
                     ['RKNE', 'RTIB'], ['RKNE', 'RTHI'], ['RASI', 'RTHI'], ['RPSI', 'RTHI'],  # "
                     ['LPSI', 'T10'], ['RPSI', 'T10'], ['LASI', 'STRN'], ['RASI', 'STRN'],  # conncet lower and upper
                     # upper
                     ['LFHD', 'LBHD'], ['RFHD', 'RBHD'], ['LFHD', 'RFHD'], ['LBHD', 'RBHD'],  # around head
                     ['LBHD', 'C7'], ['RBHD', 'C7'], ['C7', 'CLAV'], ['CLAV', 'LSHO'], ['CLAV', 'RSHO'],
                     # connect head to shoulder
                     ['LSHO', 'LBAK'], ['RSHO', 'RBAK'], ['RBAK', 'LBAK'],  # around shoulder
                     ['LWRA', 'LFIN'], ['LWRA', 'LFIN'], ['LWRA', 'LWRB'], ['LWRA', 'LFRM'], ['LWRB', 'LFRM'],
                     # around wrist
                     ['RWRA', 'RFIN'], ['RWRA', 'RFIN'], ['RWRA', 'RWRB'], ['RWRA', 'RFRM'], ['RWRB', 'RFRM'],  # "
                     ['LELB', 'LRFM'], ['LELB', 'LUPA'], ['LELB', 'LFIN'], ['LUPA', 'LSHO'],
                     # connect elbow to wrist, connect elbow to shoulder
                     ['RELB', 'RRFM'], ['RELB', 'RUPA'], ['RELB', 'RFIN'], ['RUPA', 'RSHO'],  # "
                     ['LSHO', 'STRN'], ['RSHO', 'STRN'], ['LBAK', 'T10'], ['RBAK', 'T10'],
                     # connect shoulder to torso
                     ]
            # extract initial part of joint name 'skelton 04:'hip
            jointNames = list(self.joints.keys())
            init_string = jointNames[0]
            init_string = init_string[:init_string.index(':') + 1]
            self.lines = []

            for line in lines:
                try:
                    self.lines.append(
                        [jointNames.index(init_string + line[0]), jointNames.index(init_string + line[1])])
                except:
                    continue

        else:
            print("Warning: {0} is unknown lines".format(lines))

    def show(self, fps=240, colors=None):
        if self.joints is None:
            raise NotImplementedError("show function must be implemented after setvalue or set_from_trc")
        from dp.view import Visualization
        vis = Visualization()
        data = np.array(list(self.joints.values())) # [joint index][time][dim]
        vis.show3d(x=data[:, :, 0].T, y=data[:, :, 1].T, z=data[:, :, 2].T,
                   jointNames=self.joints, lines=self.lines, fps=fps, colors=colors)

    def save(self, path, fps=240, colors=None, saveonly=True):
        if self.joints is None:
            raise NotImplementedError("save function must be implemented after setvalue or set_from_trc")
        from dp.view import Visualization
        vis = Visualization()
        data = np.array(list(self.joints.values()))  # [joint index][time][dim]
        vis.show3d(x=data[:, :, 0].T, y=data[:, :, 1].T, z=data[:, :, 2].T,
                   jointNames=self.joints, saveonly=saveonly, savepath=path, lines=self.lines, fps=fps, colors=colors)
