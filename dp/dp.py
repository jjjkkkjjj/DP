import numpy as np
from scipy.spatial.distance import cdist
import sys
sys.setrecursionlimit(10000)
import warnings
from matplotlib.colors import hsv_to_rgb
from .data import Data
from .base import DPBase
from .constraint import constraint

class DP(DPBase):
    def __init__(self, reference=None, input=None, verbose=True, ignoreWarning=False, verboseNan=True):
        """
                        @param reference, input : type Data
                        @param verbose : boolean

                        @param correspondents : dict, keys are joint names [joint name][time, ref{0}, inp{1}]
        """
        super().__init__(verbose=verbose, verboseNan=verboseNan, ignoreWarning=ignoreWarning)

        if reference is not None:
            #print(reference.__class__.__name__)
            if isinstance(reference, Data):
                self.reference = reference
            else:
                raise ValueError("you must set Data class object as reference")

            if isinstance(input, Data):
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

    def calc(self, jointNames=None, showresult=False, resultdir="", myLocalCosts=None, myMatchingCostFunc=None, correspondLine=True, returnMatchingCosts=False):
        if jointNames is None:
            jointNames = list(self.input.joints.keys()) # corresponds to input
        elif not isinstance(jointNames, list):
            raise ValueError("argument \'jointsNames\'[type:{0}] must be list or None which means calculation for all joints".format(type(jointNames).__name__))

        matchingCosts = {}

        for i, joint in enumerate(jointNames):
            if not (joint in self.reference.joints.keys() and joint in self.input.joints.keys()):
                print("\nWarning: {0} is not valid joint name".format(joint))
                continue

            refData = self.reference.joints[joint]
            inpData = self.input.joints[joint]

            if myLocalCosts is None:
                localCost = cdist(refData, inpData, 'euclidean')
            else:
                if not isinstance(myLocalCosts, dict):
                    raise ValueError("myLocalCosts must be dict")
                localCost = myLocalCosts[joint]

            correspondentPoints, matchingCost = super().calc(localCost, myMatchingCostFunc, joint)
            if matchingCost is None:
                continue

            if returnMatchingCosts:
                try:
                    tmp = np.nanargmin(matchingCost[self.reference.frame_max - 1]) # check whether all time are nan
                    matchingCosts[joint] = matchingCost
                    continue
                except ValueError:
                    continue

            self.correspondents[joint] = np.array(correspondentPoints)
            self.totalCosts[joint] = np.nanmin(matchingCost[self.reference.frame_max - 1]) / self.reference.frame_max

            if showresult:
                self.showresult(joint, correspondLine)
            if resultdir != "":
                self.saveresult(joint, savepath=resultdir + "/{0}-R_{1}-I_{2}.png".format(joint, self.reference.name, self.input.name),
                                    correspondLine=correspondLine)


        if returnMatchingCosts:
            return matchingCosts


    def calc_corrcoef(self, corrcoef, showresult=False, myMatchingCostFunc=None, resultdir="", correspondLine=True):
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

        self.calc(jointNames=jointNames, showresult=showresult, myMatchingCostFunc=myMatchingCostFunc, resultdir=resultdir, myLocalCosts=myLocalCosts, correspondLine=correspondLine)

    def calcCorrespondInitial(self, jointNames=None, showresult=False, myMatchingCostFunc=None, resultdir="", correspondLine=True):
        if jointNames is None:
            jointNames = list(self.input.joints.keys()) # corresponds to input
        elif not isinstance(jointNames, list):
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

        matchingCosts = self.calc(jointNames=jointNames, showresult=showresult, myLocalCosts=myLocalCosts, myMatchingCostFunc=myMatchingCostFunc,
                                  resultdir=resultdir, correspondLine=correspondLine, returnMatchingCosts=True)

        totalMatchingCosts = []

        for inputTime in range(self.input.frame_max):
            totalMatchingCosts.append(np.nansum([matchingCost[self.reference.frame_max - 1, inputTime] for matchingCost in matchingCosts.values()]))
        initialFrameReversed = np.argmin(totalMatchingCosts)

        if self.verbose:
            print("\ninitial frame is {0}\nback tracking now...".format(self.input.frame_max - initialFrameReversed - 1))

        for joint in matchingCosts.keys():
            correspondentPoints = np.array(myMatchingCostFunc['backTrack'](matchingCost=matchingCosts[joint], inputFinFrameBackTracked=initialFrameReversed, localCost=myLocalCosts[joint]))
            # correspondentPoints[reference, input]
            # reverse ref
            correspondentPoints[:, 0] = self.reference.frame_max - 1 - correspondentPoints[::-1, 0]
            # reverse inp
            correspondentPoints[:, 1] = self.input.frame_max - 1 - correspondentPoints[::-1, 1]

            matchingCost = matchingCosts[joint][::-1, ::-1]

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
        from dp.view import Visualization
        view = Visualization()
        view.show(x=x, y=y, xtime=self.reference.frame_max, ytime=self.input.frame_max,
                  title='Matching Path', legend=True, correspondLine=correspondLine)

    def saveresult(self, jointName, savepath, correspondLine):
        if jointName not in self.correspondents.keys():
            raise NotImplementedError("There is no result about {0}: this method must call after calc".format(jointName))
        x = {jointName: self.correspondents[jointName][:, 0]}
        y = {jointName: self.correspondents[jointName][:, 1]}

        from dp.view import Visualization
        view = Visualization()
        view.show(x=x, y=y, xtime=self.reference.frame_max, ytime=self.input.frame_max,
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

    def resultVisualization(self, kind='visualization', fps=240, maximumGapTime=0.1, resultDir=""):
        if 'visualization' not in kind:
            raise NameError('{0} is invalid kind name'.format(kind))
        myMatchingCostFunc = constraint(kind=kind)

        self.calcCorrespondInitial(showresult=False, resultdir=resultDir, myMatchingCostFunc=myMatchingCostFunc,
                                   correspondLine=True)

        return self.calc_visualization(fps=fps, maximumGapTime=maximumGapTime)

    def calc_visualization(self, fps=240, maximumGapTime=0.1):
        Ref, Inp = self.resultData()

        # colors is ndarray:[time, joint(index)]
        colors = [] # slope = 1 -> 0pt, slope = 2 -> 1pt, slope = 1/2 -> -1pt
        runningAveRange = int(fps * maximumGapTime)
        if runningAveRange < 2:
            raise ValueError('running average range(= fps*maximumGapTime) must be more than 2')

        for joint in self.input.joints.keys():
            if joint not in list(Ref.keys()):
                hsv = np.zeros((self.input.frame_max, 3))
                colors.append(hsv_to_rgb(hsv))
                continue
            ref = Ref[joint]
            inp = Inp[joint].astype('int')

            init = inp[0]
            fin = inp[-1]

            slopes = np.gradient(ref)
            v = np.ones(runningAveRange) / float(runningAveRange)
            slopes = np.convolve(slopes, v, mode='same')

            scores = slopes - 1
            scores[scores < 0] /= 0.5
            # rounding error
            scores[scores < -1] = -1
            scores[scores > 1] = 1
            #print(scores.max())
            #print(scores.min())
            """
            import matplotlib.pyplot as plt
            plt.clf()
            plt.ylim([-1, 2])
            plt.plot(np.arange(fin - init + 1), scores)
            plt.show()
            """
            # convert running average into hsv value
            # chage satulation -> white:red, blue:white
            hsvInpArea = np.zeros((scores.size, 3)) # [time, (h,s,v)]
            # red means fast
            redIndices = scores >= 0

            hsvInpArea[redIndices, 0] = 0.0
            #hsvInpArea[redIndices, 1] = scores[redIndices]
            #hsvInpArea[redIndices, 2] = 1.0
            ### center color is black
            hsvInpArea[redIndices, 1] = 1.0
            hsvInpArea[redIndices, 2] = scores[redIndices]
            # blue means slow
            blueIndices = scores < 0
            hsvInpArea[blueIndices, 0] = 2 / 3.0
            #hsvInpArea[blueIndices, 1] = np.abs(scores[blueIndices])
            #hsvInpArea[blueIndices, 2] = 1.0
            ### center color is black
            hsvInpArea[blueIndices, 1] = 1.0
            hsvInpArea[blueIndices, 2] = np.abs(scores[blueIndices])

            hsv = np.zeros((self.input.frame_max, 3))
            hsv[init:fin + 1, :] = hsvInpArea

            #print((init, fin))
            # each terminate time is difference
            colors.append(hsv_to_rgb(hsv))

        colors = np.array(colors).transpose((1, 0, 2))

        return colors

    def lowMemoryCalc(self, jointNames, showresult=False, resultdir="", myLocalCosts=None, myMatchingCostFunc=None, correspondLine=True, returnMatchingCosts=False):
        pass


    def asyncCalc(self, jointNames, showresult=False, resultdir="", myLocalCosts=None, myMatchingCostFunc=None, correspondLine=True, returnMatchingCosts=False):
        pass
        if isinstance(jointNames, list) and len(jointNames) != 2:
            raise ValueError(
                "argument \'jointsNames\'[got type:{0}] must be list and have 2 length[got len:{1}]".format(
                    type(jointNames).__name__, len(jointNames)))
        matchingCosts = {}

        joint0, joint1 = jointNames[0], jointNames[1]

        refData0 = self.reference.joints[joint0]
        inpData0 = self.input.joints[joint0]

        refData1 = self.reference.joints[joint1]
        inpData1 = self.input.joints[joint1]

        correspondentPoints, matchingCost = super().lowMemoryCalc(jointNames, myMatchingCostFunc, )
        if matchingCost is None:
            return

        if returnMatchingCosts:
            try:
                tmp = np.nanargmin(matchingCost[self.reference.frame_max - 1])  # check whether all time are nan
                matchingCosts[joint] = matchingCost
                return
            except ValueError:
                return

        self.correspondents[joint] = np.array(correspondentPoints)
        self.totalCosts[joint] = np.nanmin(matchingCost[self.reference.frame_max - 1]) / self.reference.frame_max

        if showresult:
            self.showresult(joint, correspondLine)
        if resultdir != "":
            self.saveresult(joint, savepath=resultdir + "/{0}-R_{1}-I_{2}.png".format(joint, self.reference.name,
                                                                                      self.input.name),
                            correspondLine=correspondLine)

        if returnMatchingCosts:
            return matchingCosts