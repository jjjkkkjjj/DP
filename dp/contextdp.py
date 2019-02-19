from .dp import DP
from scipy.spatial.distance import cdist
import numpy as np
from .constraint import constraint, lowMemoryConstraint

class SyncContextDP(DP):
    def __init__(self, contexts, **kwargs):
        super().__init__(**kwargs)

        self.contexts = contexts

    def synchronous(self, myMatchingCostFunc=None, myLocalCosts=None, returnMatchingCosts=False):
        self.correspondents = {}

        matchingCosts = {}
        for index, context in enumerate(self.contexts):
            contextKey = ''
            for joint in self.contexts[index]:
                contextKey += joint
                contextKey += '-'
            contextKey = contextKey[:-1]

            if myLocalCosts is None:
                localCost = np.zeros((self.reference.frame_max, self.input.frame_max))
                for joint in context:
                    localCost += cdist(self.reference.joints[joint], self.input.joints[joint], 'euclidean')

            elif isinstance(myLocalCosts, dict):
                localCost = myLocalCosts[contextKey]
            else:
                raise ValueError('myLocalCosts must be list')
            # call method in super of super class which is DPBase
            self.correspondents[contextKey], matchingCosts[contextKey] = super(DP, self).calc(localCost, myMatchingCostFunc)
            self.correspondents[contextKey] = np.array(self.correspondents[contextKey])
        if returnMatchingCosts:
            return matchingCosts

    def syncCorrespondInitial(self, myMatchingCostFunc=None):
        myLocalCosts = {}
        for index, context in enumerate(self.contexts):
            localCost = np.zeros((self.reference.frame_max, self.input.frame_max))
            for joint in context:
                localCost += cdist(self.reference.joints[joint][::-1, :],
                                   self.input.joints[joint][::-1, :], 'euclidean')
            localCost /= len(context)

            contextKey = ''
            for joint in self.contexts[index]:
                contextKey += joint
                contextKey += '-'
            contextKey = contextKey[:-1]
            myLocalCosts[contextKey] = localCost
        matchingCosts = self.synchronous(myMatchingCostFunc=myMatchingCostFunc, myLocalCosts=myLocalCosts,
                                  returnMatchingCosts=True)
        self._searchInitialFrame(matchingCosts, myMatchingCostFunc, myLocalCosts)


    def _searchInitialFrame(self, matchingCosts, myMatchingCostFunc, myLocalCosts):

        if self.verbose:
            print("calculating matching costs...")

        totalMatchingCosts = []

        for inputTime in range(self.input.frame_max):
            totalMatchingCosts.append(np.nansum(
                [matchingCost[self.reference.frame_max - 1, inputTime] for matchingCost in matchingCosts.values()]))
        initialFrameReversed = np.argmin(totalMatchingCosts)

        if self.verbose:
            print("\ninitial frame is {0}\nback tracking now...".format(
                self.input.frame_max - initialFrameReversed - 1))

        backTrackFunc = myMatchingCostFunc['backTrack']
        for contextKey in matchingCosts.keys():
            correspondentPoints = np.array(
                backTrackFunc(matchingCost=matchingCosts[contextKey], inputFinFrameBackTracked=initialFrameReversed,
                        localCost=myLocalCosts[contextKey]))
            # correspondentPoints[reference, input]
            # reverse ref
            correspondentPoints[:, 0] = self.reference.frame_max - 1 - correspondentPoints[::-1, 0]
            # reverse inp
            correspondentPoints[:, 1] = self.input.frame_max - 1 - correspondentPoints[::-1, 1]

            matchingCost = matchingCosts[contextKey][::-1, ::-1]

            self.correspondents[contextKey] = correspondentPoints
            self.totalCosts[contextKey] = np.nanmin(matchingCost[self.reference.frame_max - 1]) / self.reference.frame_max

    def resultVisualization(self, kind='visualization', fps=240, maximumGapTime=0.1, resultDir="", **kwargs):
        if 'visualization' not in kind and 'localdiff' not in kind:
            raise NameError('{0} is invalid kind name'.format(kind))
        # calc sync dp for each context
        myMatchingCostFunc = constraint(kind)
        self.syncCorrespondInitial(myMatchingCostFunc=myMatchingCostFunc)

        # revert context into each joint
        newcorrespondents = {}
        for contextKey, correspondent in self.correspondents.items():
            for joint in contextKey.split('-'):
                newcorrespondents[joint] = correspondent
        self.correspondents = newcorrespondents

        # solve colormap of timing gaps
        if 'visualization' in kind:
            return super().calc_visualization(fps=fps, maximumGapTime=maximumGapTime)
        else: # localdiff
            return super().calc_visualization_localdiff(fps=fps, maximumGapTime=maximumGapTime)

    def resultData(self):
        if len(self.correspondents) == 0:
            raise NotImplementedError("There is no result: this method must call after calc")

        x = {}
        y = {}
        for contextKey, correspondent in self.correspondents.items():
            x[contextKey] = correspondent[:, 0]
            y[contextKey] = correspondent[:, 1]

        return x, y


class AsyncContextDP(DP):
    def __init__(self, contexts, **kwargs):
        super().__init__(**kwargs)

        self.contexts = contexts

    def asynchronous(self, kinds, returnMatchingCosts=False):
        if not isinstance(kinds, list) and len(kinds) != len(self.contexts):
            raise ValueError('myMatchingCostFuncs must be list and have same length to contexts')
        self.correspondents = {}

        matchingCosts = {}
        for index, context in enumerate(self.contexts):
            contextKey = ''
            for joint in self.contexts[index]:
                contextKey += joint
                contextKey += '-'
            contextKey = contextKey[:-1]

            refDatas, inpDatas = [], []
            for joint in context:
                refDatas.append(self.reference.joints[joint])
                inpDatas.append(self.input.joints[joint])

            # call method in super of super class which is DPBase
            correspondentPointses, matchingCosts[contextKey] =\
                super(DP, self).lowMemoryCalc(refDatas, inpDatas, myMatchingCostFunc=lowMemoryConstraint(kinds[index]), name=contextKey)
            for correspondentPoints, joint in zip(correspondentPointses, self.contexts[index]):
                self.correspondents[joint] = np.array(correspondentPoints)

        if returnMatchingCosts:
            return matchingCosts

    def asyncCorrespondInitial(self, kinds):
        for index, context in enumerate(self.contexts):
            for joint in self.contexts[index]:
                self.reference.joints[joint] = self.reference.joints[joint][::-1, :]
                self.input.joints[joint] = self.input.joints[joint][::-1, :]

        matchingCosts = self.asynchronous(kinds, returnMatchingCosts=True)
        self._searchInitialFrame(matchingCosts, kinds)

        # revert order
        for index, context in enumerate(self.contexts):
            for joint in self.contexts[index]:
                self.reference.joints[joint] = self.reference.joints[joint][::-1, :]
                self.input.joints[joint] = self.input.joints[joint][::-1, :]

    def _searchInitialFrame(self, matchingCosts, kinds):

        if self.verbose:
            print("calculating matching costs...")

        totalMatchingCosts = []

        limits = 2
        for inputTime in range(self.input.frame_max):
            tmp = 0
            for contextKey, matchingCost in matchingCosts.items():
                contextNum = len(contextKey.split('-'))
                if contextNum == 2:
                    tmp += matchingCost[self.reference.frame_max - 1, inputTime, 2*limits] / contextNum
                elif contextNum == 3:
                    tmp += matchingCost[self.reference.frame_max - 1, inputTime, 2*limits, 2*limits] / contextNum
            totalMatchingCosts.append(tmp)
        initialFrameReversed = np.argmin(totalMatchingCosts)

        if self.verbose:
            print("\ninitial frame is {0}\nback tracking now...".format(
                self.input.frame_max - initialFrameReversed - 1))

        for contextKey, kind in zip(matchingCosts.keys(), kinds):
            backTrackFunc = lowMemoryConstraint(kind)['backTrack']
            correspondentPointses = np.array(
                backTrackFunc(matchingCost=matchingCosts[contextKey], inputFinFrameBackTracked=initialFrameReversed))

            for correspondentPoints, joint in zip(correspondentPointses, contextKey.split('-')):
                correspondentPoints = np.array(correspondentPoints)
                # correspondentPoints[reference, input]
                # reverse ref
                correspondentPoints[:, 0] = self.reference.frame_max - 1 - correspondentPoints[::-1, 0]
                # reverse inp
                correspondentPoints[:, 1] = self.input.frame_max - 1 - correspondentPoints[::-1, 1]

                self.correspondents[joint] = np.array(correspondentPoints)

                matchingCost = matchingCosts[contextKey][::-1, ::-1]
                self.totalCosts[joint] = np.nanmin(
                    matchingCost[self.reference.frame_max - 1]) / self.reference.frame_max


    def resultVisualization(self, kind='visualization2', fps=240, maximumGapTime=0.1, resultDir="", **kwargs):
        if 'visualization' not in kind and 'localdiff' not in kind:
            raise NameError('{0} is invalid kind name'.format(kind))
        try:
            kinds = kwargs.pop('kinds')
            for kind_ in kinds:
                if kind not in kind_ and kind not in kind_:
                    raise ValueError('all kinds must be included \'visualization2\' but got {0}'.format(kind_))
            self.asyncCorrespondInitial(kinds)
        except KeyError:
            raise NameError('kinds must be set as argument')

        # revert context into each joint
        newcorrespondents = {}
        for contextKey, correspondent in self.correspondents.items():
            for joint in contextKey.split('-'):
                newcorrespondents[joint] = correspondent
        self.correspondents = newcorrespondents

        # solve colormap of timing gaps
        if 'visualization' in kind:
            return super().calc_visualization(fps=fps, maximumGapTime=maximumGapTime)
        else: # localdiff
            return super().calc_visualization_localdiff(fps=fps, maximumGapTime=maximumGapTime)

    def resultData(self):
        if len(self.correspondents) == 0:
            raise NotImplementedError("There is no result: this method must call after calc")

        x = {}
        y = {}
        for contextKey, correspondent in self.correspondents.items():
            x[contextKey] = correspondent[:, 0]
            y[contextKey] = correspondent[:, 1]

        return x, y

