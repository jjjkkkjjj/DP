import numpy as np
import csv
from scipy.interpolate import CubicSpline as cs
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
import sys
sys.setrecursionlimit(10000)
import os
from dp.view import Visualization
import warnings


class DP(Visualization):
    def __init__(self, reference=None, input=None, verbose=True, ignoreWarning=False, verboseNan=True):
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
        if ignoreWarning:
            np.seterr(invalid='ignore')
            warnings.filterwarnings("ignore")
        self.verbose = verbose
        self.verboseNan = verboseNan
        self.correspondents = {}
        self.totalCosts = {}
        self.matchingCosts = {}

    def calc(self, jointNames=None, showresult=False, resultdir="", myLocalCosts=None, correspondLine=True, matchingcost=False):
        if jointNames is None:
            jointNames = list(self.input.joints.keys()) # corresponds to input
        elif type(jointNames).__name__ != 'list':
            raise ValueError("argument \'jointsNames\'[type:{0}] must be list or None which means calculation for all joints".format(type(jointNames).__name__))

        for joint in jointNames:
            if not (joint in self.reference.joints.keys() and joint in self.input.joints.keys()):
                print("Warning: {0} is not valid joint name".format(joint))
                continue

            if self.verbose:
                sys.stdout.write("\r{0} is calculating...".format(joint))
                sys.stdout.flush()

            refData = self.reference.joints[joint]
            inpData = self.input.joints[joint]

            if myLocalCosts is None:
                localCosts = cdist(refData, inpData, 'euclidean')
            else:
                if type(myLocalCosts).__name__ != 'dict':
                    raise ValueError("myLocalCosts must be dict")
                localCosts = myLocalCosts[joint]

            matchingCost = np.zeros(localCosts.shape)
            matchingCost[0, :] = localCosts[0, :]

            for referenceTime in range(1, self.reference.frame_max):
                matchingCost[referenceTime, 0] = localCosts[referenceTime, 0] + matchingCost[referenceTime - 1, 0]
                matchingCost[referenceTime, 1] = localCosts[referenceTime, 1] + np.minimum(matchingCost[referenceTime - 1, 0],
                                                                                            matchingCost[referenceTime - 1, 1])
                matchingCost[referenceTime, 2:] = localCosts[referenceTime, 2:] + np.minimum.reduce([matchingCost[referenceTime - 1, 2:],
                                                                                                      matchingCost[referenceTime - 1, 1:-1],
                                                                                                      matchingCost[referenceTime - 1, :-2]])

            if matchingcost:
                try:
                    tmp = np.nanargmin(matchingCost[self.reference.frame_max - 1]) # check whether all time are nan
                    self.matchingCosts[joint] = matchingCost
                    continue
                except ValueError:
                    continue

            # back track
            try:
                correspondentPoints = self.__backTrack(matchingCost, inputFinFrame=np.nanargmin(matchingCost[self.reference.frame_max - 1]))

                self.correspondents[joint] = np.array(correspondentPoints)
                self.totalCosts[joint] = np.nanmin(
                    matchingCost[self.reference.frame_max - 1]) / self.reference.frame_max

                if self.verbose:
                    sys.stdout.write("\r{0} is calculating...finished\n".format(joint))
                    sys.stdout.flush()
                if showresult:
                    self.showresult(joint, correspondLine)
                if resultdir != "":
                    self.saveresult(joint,
                                    savepath=resultdir + "/{0}-R_{1}-I_{2}.png".format(joint, self.reference.name,
                                                                                       self.input.name),
                                    correspondLine=correspondLine)

            except ValueError:
                #if self.verbose:
                if self.verboseNan:
                    sys.stdout.write("\rWarning:{0}'s all matching cost has nan\nskip...\n".format(joint))
                    sys.stdout.flush()
                continue


    def calc_corrcoef(self, corrcoef, showresult=False, resultdir="", correspondLine=True):
        jointNames = corrcoef['jointNames']
        Neighbors = corrcoef['neighbor']
        Corrcoefs = corrcoef['corrcoef']
        IgnoredJoints = corrcoef['ignored']

        # unique jointnames between ref and inp
        #if self.reference.joints.keys() != self.input.joints.keys():
        #    raise ValueError("The joints of reference and input must be same")
        #if self.reference.lines != self.input.lines:
        #    raise ValueError("The lines of reference and input must be same")


        myLocalCosts = {}
        LocalCosts = [cdist(self.reference.joints[joint], self.input.joints[joint], 'euclidean') for joint in jointNames]
        # print(type(LocalCosts[index])) ndarray
        # np.array(LocalCosts)[neighbors].transpose(1, 2, 0) is converting (refT,inpT,Cnum) into (Cnum,refT,inpT) for inner product to corrcoef
        for index, joint in enumerate(jointNames):
            if joint in IgnoredJoints:
                myLocalCosts[joint] = LocalCosts[index]
                continue
            myLocalCosts[joint] = LocalCosts[index] + np.inner(Corrcoefs[joint], np.array(LocalCosts)[Neighbors[joint]].transpose(1, 2, 0)) \
                                                      / (1 + np.sum(Corrcoefs[joint]))

            #myLocalCosts[joint] = LocalCosts[index]

        self.calc(jointNames=jointNames, showresult=showresult, resultdir=resultdir, myLocalCosts=myLocalCosts, correspondLine=correspondLine)

    def calcCorrespondInitial(self, jointNames=None, showresult=False, resultdir="", correspondLine=True):
        if jointNames is None:
            jointNames = list(self.input.joints.keys()) # corresponds to input
        elif type(jointNames).__name__ != 'list':
            raise ValueError("argument \'jointsNames\'[type:{0}] must be list or None which means calculation for all joints".format(type(jointNames).__name__))
        elif len(jointNames) == 1:
            print("Warning: jointNames\' length was 1, this result will be same to calc")
            self.calc(jointNames=jointNames, showresult=showresult, resultdir=resultdir, correspondLine=correspondLine)
            return

        myLocalCosts = {}
        for joint in jointNames:
            if not (joint in self.reference.joints.keys() and joint in self.input.joints.keys()):
                print("Warning: {0} is not valid joint name".format(joint))
                continue

            refData = self.reference.joints[joint]
            inpData = self.input.joints[joint]

            #print(refData.shape)  (295, 3)=(time,dim)
            # reverse time
            refData = refData[::-1,:]
            inpData = inpData[::-1,:]

            myLocalCosts[joint] = cdist(refData, inpData, 'euclidean')

        if self.verbose:
            print("calculating matching costs...")

        self.calc(jointNames=jointNames, showresult=showresult, myLocalCosts=myLocalCosts, resultdir=resultdir, correspondLine=correspondLine, matchingcost=True)

        totalMatchingCosts = []

        for inputTime in range(self.input.frame_max):
            totalMatchingCosts.append(np.sum([matchingCost[self.reference.frame_max - 1, inputTime] for matchingCost in self.matchingCosts.values()]))
        initialFrameReversed = np.argmin(totalMatchingCosts)

        if self.verbose:
            print("\ninitial frame is {0}\nback tracking now...".format(self.input.frame_max - initialFrameReversed - 1))

        for joint in self.matchingCosts.keys():
            correspondentPoints = np.array(self.__backTrack(self.matchingCosts[joint], initialFrameReversed))
            # correspondentPoints[reference, input]
            # reverse ref
            correspondentPoints[:, 0] = self.reference.frame_max - 1 - correspondentPoints[::-1, 0]
            # reverse inp
            correspondentPoints[:, 1] = self.input.frame_max - 1 - correspondentPoints[::-1, 1]

            matchingCost = self.matchingCosts[joint][::-1, ::-1]

            self.correspondents[joint] = correspondentPoints
            self.totalCosts[joint] = np.nanmin(
                matchingCost[self.reference.frame_max - 1]) / self.reference.frame_max

            if self.verbose:
                sys.stdout.write("\r{0} is calculating...finished\n".format(joint))
                sys.stdout.flush()
            if showresult:
                self.showresult(joint, correspondLine)
            if resultdir != "":
                self.saveresult(joint,
                                savepath=resultdir + "/{0}-R_{1}-I_{2}.png".format(joint, self.reference.name,
                                                                                   self.input.name),
                                correspondLine=correspondLine)

        self.matchingCosts = {}


    def __backTrack(self, matchingCost, inputFinFrame):
        # back track
        correspondentPoints = []
        r, i = self.reference.frame_max - 1, inputFinFrame
        correspondentPoints.append([r, i])

        while r > 0 and i > 2:
            tmp = np.argmin((matchingCost[r - 1, i], matchingCost[r - 1, i - 1],
                             matchingCost[r - 1, i - 2]))
            r = r - 1
            i = i - tmp
            correspondentPoints.insert(0, [r, i])

        while r > 0 and i > 1:
            tmp = np.argmin((matchingCost[r - 1, i], matchingCost[r - 1, i - 1]))
            r = r - 1
            i = i - tmp
            correspondentPoints.insert(0, [r, i])

        while r > 0:
            r = r - 1
            i = 0
            correspondentPoints.insert(0, [r, i])

        return correspondentPoints

    def aligned(self, jointNames=None): # input aligned by reference
        if jointNames is None:
            jointNames = self.input.joints

        aligned = {}

        for jointName in jointNames:
            if jointName not in self.correspondents.keys():
                raise NotImplementedError("There is no result about {0}: this method must call after calc".format(jointName))

            times = self.correspondents[jointName][:, 1]
            aligned[jointName] = self.input.joints[jointName][times, :]

        return aligned

    def showresult(self, jointName, correspondLine):
        if jointName not in self.correspondents.keys():
            raise NotImplementedError("There is no result about {0}: this method must call after calc".format(jointName))
        x = {jointName: self.correspondents[jointName][:, 0]}
        y = {jointName: self.correspondents[jointName][:, 1]}

        self.show(x=x, y=y, xtime=self.reference.frame_max, ytime=self.input.frame_max,
                  title='Matching Path', legend=True, correspondLine=correspondLine)

    def saveresult(self, jointName, savepath, correspondLine):
        if jointName not in self.correspondents.keys():
            raise NotImplementedError("There is no result about {0}: this method must call after calc".format(jointName))
        x = {jointName: self.correspondents[jointName][:, 0]}
        y = {jointName: self.correspondents[jointName][:, 1]}

        self.show(x=x, y=y, xtime=self.reference.frame_max, ytime=self.input.frame_max,
                  title='Matching Path', legend=True, savepath=savepath, correspondLine=correspondLine, verbose=self.verbose)

    def resultData(self):
        if len(self.correspondents) == 0:
            raise NotImplementedError("There is no result: this method must call after calc")

        x = {}
        y = {}
        for joint, correspondent in self.correspondents.items():
            x[joint] = correspondent[:, 0]
            y[joint] = correspondent[:, 1]

        return x, y

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

    def show(self, fps=240):
        if self.joints is None:
            raise NotImplementedError("show function must be implemented after setvalue or set_from_trc")
        vis = Visualization()
        data = np.array(list(self.joints.values())) # [joint index][time][dim]
        vis.show3d(x=data[:, :, 0].T, y=data[:, :, 1].T, z=data[:, :, 2].T, jointNames=self.joints, lines=self.lines, fps=fps)

    def save(self, path, fps=240, saveonly=True):
        if self.joints is None:
            raise NotImplementedError("save function must be implemented after setvalue or set_from_trc")
        vis = Visualization()
        data = np.array(list(self.joints.values()))  # [joint index][time][dim]
        vis.show3d(x=data[:, :, 0].T, y=data[:, :, 1].T, z=data[:, :, 2].T, jointNames=self.joints, saveonly=saveonly, savepath=path, lines=self.lines, fps=fps)

def referenceDetector(Datalists, name, save=True, superDir='/', verbose=False, verboseNan=False):
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

                dp = DP(dataRef, dataInp, verbose=verbose, ignoreWarning=True, verboseNan=verboseNan)
                dp.calc(showresult=False)
                Matrix_DPcost[row][col] = np.mean(list(dp.totalCosts.values()))


                del dp
        print("\nfinished")
        print(Matrix_DPcost)
        score = np.array(
            [np.sum(Matrix_DPcost[:][index]) + np.sum(Matrix_DPcost[index][:]) for index in range(len(Datalists))])
        centroidIndex = np.argmin(score)
        print("reference is \"{0}\"".format(Datalists[centroidIndex].name))

        if save:
            savepath = os.path.join('./reference/', superDir)
            if not os.path.exists(savepath):
                os.mkdir(savepath)
            savepath = os.path.join(savepath, name)
            with open(savepath, 'a') as f_ref:
                if Datalists[centroidIndex].dir is None:
                    f_ref.write(
                        'dir,None,the number of files,{0},optimal reference,{1},\n'.format(len(Datalists), Datalists[centroidIndex].name))
                else:
                    f_ref.write(
                        'dir,{0},the number of files,{1},optimal reference,{2},\n'.format(Datalists[centroidIndex].dir,
                                                                                          len(Datalists),
                                                                                          Datalists[centroidIndex].name))
                np.savetxt(f_ref, Matrix_DPcost, delimiter=',')
            print('saved {0}'.format(savepath))

def referenceReader(name, directory, superDir='/'):
    referenceCsvPath = os.path.join('./reference/', superDir)
    if not os.path.exists(referenceCsvPath):
        raise IsADirectoryError("{0} doesn\'t exist".format(referenceCsvPath))
    with open(os.path.join(referenceCsvPath, name), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == 'dir':
                if row[1] == directory:
                    return row[5]
    raise ValueError("No such a {0}".format(directory))
