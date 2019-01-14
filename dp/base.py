import numpy as np
import sys
import warnings
from .constraint import constraint

class DPBase():
    def  __init__(self, verbose=False, verboseNan=False, ignoreWarning=True):
        self.verbose = verbose
        self.verboseNan = verboseNan

        if ignoreWarning:
            np.seterr(invalid='ignore')
            warnings.filterwarnings("ignore")

    def calc(self, localCost, myMatchingCostFunc=None):
        if myMatchingCostFunc is None:
            myMatchingCostFunc = constraint('default')
            matchingCostFunc = myMatchingCostFunc['matchingCost']
            backTrackFunc = myMatchingCostFunc['backTrack']
        elif not isinstance(myMatchingCostFunc, dict):
            try:
                matchingCostFunc = myMatchingCostFunc['matchingCost']
                backTrackFunc = myMatchingCostFunc['backTrack']
            except KeyError:
                raise KeyError('myMatchingCostFunc must be dict, and one\'s key must have [\'matchingCost\',\'backTrack\']')
        else:
            raise ValueError('myMatchingCostFunc must be dict, and one\'s key must have [\'matchingCost\',\'backTrack\']')


        try:
            matchingCost = matchingCostFunc(localCost)
        except:
            if self.verbose:
                sys.stdout.write("\rWarning:{0}\nskip...\n".format(sys.exc_info()))
                sys.stdout.flush()
                return None, None

        try:
            correspondentPoints = backTrackFunc(matchingCost=matchingCost, localCost=localCost,
                                                inputFinFrameBackTracked=np.nanargmin(
                                                    matchingCost[matchingCost.shape[0] - 1]))
        except ValueError:
            # if self.verbose:
            if self.verboseNan:
                sys.stdout.write("\rWarning:all matching cost has nan\nskip...\n")
                sys.stdout.flush()
            return None, None

        return correspondentPoints, matchingCost



    def save(self, **kwargs):
        pass
