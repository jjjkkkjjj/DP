from .base import DPBase
from .dp import DP
from scipy.spatial.distance import cdist
import numpy as np
from .data import Data

class ContextDP(DP):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.contexts = None

    def sync(self, contexts, myMatchingCostFunc=None, myLocalCosts=None, returnMatchingCosts=False):
        self.contexts = contexts
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

    def syncCorrespondInitial(self, contexts, myMatchingCostFunc=None):
        myLocalCosts = {}
        for index, context in enumerate(self.contexts):
            localCost = np.zeros((self.reference.frame_max, self.input.frame_max))
            for joint in context:
                localCost += cdist(self.reference.joints[joint], self.input.joints[joint], 'euclidean')

            contextKey = ''
            for joint in self.contexts[index]:
                contextKey += joint
                contextKey += '-'
            contextKey = contextKey[:-1]
            myLocalCosts[contextKey] = localCost
        matchingCosts = self.sync(contexts, myMatchingCostFunc=myMatchingCostFunc, myLocalCosts=myLocalCosts,
                                  returnMatchingCosts=True)
        self._searchInitialFrame(matchingCosts, myMatchingCostFunc, myLocalCosts)


    def _searchInitialFrame(self, matchingCosts, myMatchingCostFunc, myLocalCosts):

        if self.verbose:
            print("calculating matching costs...")

        totalMatchingCosts = []
        exit()
        for inputTime in range(self.input.frame_max):
            totalMatchingCosts.append(np.nansum(
                [matchingCost[self.reference.frame_max - 1, inputTime] for matchingCost in matchingCosts.values()]))
        initialFrameReversed = np.argmin(totalMatchingCosts)

        if self.verbose:
            print("\ninitial frame is {0}\nback tracking now...".format(
                self.input.frame_max - initialFrameReversed - 1))

        backTrackFunc = myMatchingCostFunc['backTrack']
        for joint in matchingCosts.keys():
            correspondentPoints = np.array(
                backTrackFunc(matchingCost=matchingCosts[joint], inputFinFrameBackTracked=initialFrameReversed,
                        localCost=myLocalCosts[joint]))
            # correspondentPoints[reference, input]
            # reverse ref
            correspondentPoints[:, 0] = self.reference.frame_max - 1 - correspondentPoints[::-1, 0]
            # reverse inp
            correspondentPoints[:, 1] = self.input.frame_max - 1 - correspondentPoints[::-1, 1]

            matchingCost = matchingCosts[joint][::-1, ::-1]

            self.correspondents[joint] = correspondentPoints
            self.totalCosts[joint] = np.nanmin(matchingCost[self.reference.frame_max - 1]) / self.reference.frame_max


    def resultData(self):
        if len(self.correspondents) == 0:
            raise NotImplementedError("There is no result: this method must call after calc")

        x = {}
        y = {}
        for contextKey, correspondent in self.correspondents.items():
            x[contextKey] = correspondent[:, 0]
            y[contextKey] = correspondent[:, 1]

        return x, y