import numpy as np
import csv
from scipy.interpolate import CubicSpline as cs
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
import sys
from view import Visualization
import os


class DP(Visualization):
    def __init__(self, reference=None, input=None, verbose=True, ignoreWarning=False):
        """
                        @param reference, input : type Data
                        @param verbose : boolean

                        @param correspondents : dict, keys are joint names [joint name][time, ref{0}, inp{1}]
        """
        super(DP, self).__init__()

        if reference is not None:
            #print(reference.__class__.__name__)
            if reference.__class__.__name__ == 'Data':
                self.reference = reference
            else:
                raise ValueError("you must set Data class object as reference")

            if reference.__class__.__name__ == 'Data':
                self.input = input
            else:
                raise ValueError("you must set Data class object as input")

        if reference.frame_max > input.frame_max and not ignoreWarning:
            print("Warning: reference pattern[t:{0}] has loneger times than input one[t:{1}]".format(reference.frame_max, input.frame_max))
            #exit()

        self.verbose = verbose
        self.correspondents = {}
        self.totalCosts = {}


    def calc(self, jointNames=None, showresult=False, resultdir=""):
        if jointNames is None:
            jointNames = self.input.joints # corresponds to input
        elif type(jointNames).__name__ != 'list':
            raise ValueError("argument \'joints\'[type:{0}] must be list or None which means calculation for all joints".format(type(joints).__name__))

        for joint in jointNames:
            if not (joint in self.reference.joints.keys() and joint in self.input.joints.keys()):
                print("Warning: {0} is not valid joint name".format(joint))
                continue

            if self.verbose:
                sys.stdout.write("\r{0} is calculating...".format(joint))
                sys.stdout.flush()

            refData = self.reference.joints[joint]
            inpData = self.input.joints[joint]

            localCosts = cdist(refData, inpData, 'euclidean')

            matchingCosts = np.zeros(localCosts.shape)
            matchingCosts[0, :] = localCosts[0, :]

            for referenceTime in range(1, self.reference.frame_max):
                matchingCosts[referenceTime, 0] = localCosts[referenceTime, 0] + matchingCosts[referenceTime - 1, 0]
                matchingCosts[referenceTime, 1] = localCosts[referenceTime, 1] + np.minimum(matchingCosts[referenceTime - 1, 0],
                                                                                            matchingCosts[referenceTime - 1, 1])
                matchingCosts[referenceTime, 2:] = localCosts[referenceTime, 2:] + np.minimum.reduce([matchingCosts[referenceTime - 1, 2:],
                                                                                                      matchingCosts[referenceTime - 1, 1:-1],
                                                                                                      matchingCosts[referenceTime - 1, :-2]])

            # back track
            correspondentPoints = []
            try:
                r, i = self.reference.frame_max - 1, np.nanargmin(matchingCosts[self.reference.frame_max - 1])
                correspondentPoints.append([r, i])

                data_ = {}
                data_['tmc'] = matchingCosts[r][i]

                while r > 0 and i > 2:
                    tmp = np.argmin((matchingCosts[r - 1, i], matchingCosts[r - 1, i - 1],
                                     matchingCosts[r - 1, i - 2]))
                    r = r - 1
                    i = i - tmp
                    correspondentPoints.insert(0, [r, i])

                    """
                    if tmp == 0:
                        r = r - 1
                        i = i
                    elif tmp == 1:
                        r = r - 1
                        i = i - 1
                    else:
                        r = r -1
                        i = i - 2
                    """

                while r > 0 and i > 1:
                    tmp = np.argmin((matchingCosts[r - 1, i], matchingCosts[r - 1, i - 1]))
                    r = r - 1
                    i = i - tmp
                    correspondentPoints.insert(0, [r, i])

                while r > 0:
                    r = r - 1
                    i = 0
                    correspondentPoints.insert(0, [r, i])

                self.correspondents[joint] = np.array(correspondentPoints)
                self.totalCosts[joint] = np.nanmin(matchingCosts[self.reference.frame_max - 1]) / self.reference.frame_max

                if self.verbose:
                    sys.stdout.write("\r{0} is calculating...finished\n".format(joint))
                    sys.stdout.flush()
                if showresult:
                    self.showresult(joint)
                if resultdir != "":
                    self.saveresult(joint, savepath=resultdir + "/{0}-R_{1}-I_{2}.png".format(joint, self.reference.name, self.input.name))

            except ValueError:
                if self.verbose:
                    print("All matching cost have nan")
                    print("skip...")
                continue

    def showresult(self, jointName):
        if jointName not in self.correspondents.keys():
            raise NotImplementedError("There is no result about {0}: this method must call after calc".format(jointName))
        self.show(x=self.correspondents[jointName][:, 0], y=self.correspondents[jointName][:, 1],
                  xtime=self.reference.frame_max, ytime=self.input.frame_max, title=jointName)

    def saveresult(self, jointName, savepath):
        if jointName not in self.correspondents.keys():
            raise NotImplementedError("There is no result about {0}: this method must call after calc".format(jointName))
        self.show(x=self.correspondents[jointName][:, 0], y=self.correspondents[jointName][:, 1],
                  xtime=self.reference.frame_max, ytime=self.input.frame_max, title=jointName, savepath=savepath)

class Data:
    def __init__(self, interpolate='linear', ):
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

        pass

    def set_from_trc(self, path):
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

    def setvalues(self, dataname, x, y, z, jointNames, jointaxis=0, dir=None):
        if type(x).__name__ != 'ndarray' or type(y).__name__ != 'ndarray' or type(z).__name__ != 'ndarray':
            raise TypeError("x, y, z must be ndarray")

        if type(jointNames).__name__ != 'list':
            raise TypeError("joitNames must be list")

        if x.shape != y.shape or y.shape != z.shape or z.shape != y.shape:
            raise ValueError("x, y, z must have same shape")

        if x.shape[jointaxis] != len(jointNames):
            raise ValueError("data shape[{0}] and jointNames[size:0] must be same".format(x.shape[jointaxis], len(jointNames)))

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

def referenceDetector(Datalists, name, save=True):
    if type(Datalists).__name__ != 'list':
        raise TypeError("Datalists must be list")
    if len(Datalists) == 0:
        print("Warning: Datalists has no element")
        return -1
    else:
        if Datalists[0].__class__.__name__ != 'Data':
            raise TypeError("Datalists\' element must be \'Data\'")


        print("calculate centroid pattern")
        Matrix_DPcost = np.zeros((len(Datalists), len(Datalists)))

        for row, dataRef in enumerate(Datalists):
            sys.stdout.write('\rcalculating now... {0} / {1}'.format(row, len(Datalists)))
            sys.stdout.flush()
            for col, dataInp in enumerate(Datalists):
                if row == col:
                    continue

                dp = DP(dataRef, dataInp, verbose=False, ignoreWarning=True)
                dp.calc(showresult=False)
                Matrix_DPcost[row][col] = np.mean(dp.totalCosts.values())


                del dp
        print("\nfinished")
        print(Matrix_DPcost)
        score = np.array(
            [np.sum(Matrix_DPcost[:][index]) + np.sum(Matrix_DPcost[index][:]) for index in range(len(Datalists))])
        centroidIndex = np.argmin(score)
        print("reference is \"{0}\"".format(Datalists[centroidIndex].name))

        if save:
            with open('./reference/{0}'.format(name), 'a') as f_ref:
                if Datalists[centroidIndex].dir is None:
                    f_ref.write(
                        'dir,None,the number of files,{0},optimal reference,{1},\n'.format(len(Datalists), Datalists[centroidIndex].name))
                else:
                    f_ref.write(
                        'dir,{0},the number of files,{1},optimal reference,{2},\n'.format(Datalists[centroidIndex].dir,
                                                                                          len(Datalists),
                                                                                          Datalists[centroidIndex].name))
                np.savetxt(f_ref, Matrix_DPcost, delimiter=',')
            print('saved ./reference/{0}'.format(name))

def referenceReader(name, directory):
    with open('./reference/{0}'.format(name), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == 'dir':
                if row[1] == directory:
                    return row[7]
    raise ValueError("No such a {0}".format(directory))
