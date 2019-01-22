import numpy as np
import sys
import warnings
from .constraint import constraint

class DPBase(object):
    def  __init__(self, verbose=False, verboseNan=False, ignoreWarning=True):
        self.verbose = verbose
        self.verboseNan = verboseNan

        if ignoreWarning:
            np.seterr(invalid='ignore')
            warnings.filterwarnings("ignore")

    def calc(self, localCost, myMatchingCostFunc=None, name=''):
        if self.verbose:
            sys.stdout.write("\r{0} is calculating...".format(name))
            sys.stdout.flush()

        matchingCostFunc, backTrackFunc = check_myMatchingCostFunc(myMatchingCostFunc)

        try:
            matchingCost = matchingCostFunc(localCost)
        except:
            if self.verbose:
                sys.stdout.write("\rWarning:{0}:{1}\nskip...\n".format(name, sys.exc_info()))
                sys.stdout.flush()
                return None, None

        try:
            correspondentPoints = backTrackFunc(matchingCost=matchingCost, localCost=localCost,
                                                inputFinFrameBackTracked=np.nanargmin(
                                                    matchingCost[matchingCost.shape[0] - 1]))
        except ValueError:
            # if self.verbose:
            if self.verboseNan:
                sys.stdout.write("\rWarning:{0}\'s all matching cost has nan\nskip...\n".format(name))
                sys.stdout.flush()
            return None, matchingCost

        if self.verbose:
            sys.stdout.write("\r{0} is calculating...finished\n".format(name))
            sys.stdout.flush()

        return correspondentPoints, matchingCost

    def lowMemoryCalc(self, refDatas, inpDatas, myMatchingCostFunc=None, name=''):
        check_args(refDatas, inpDatas)

        if self.verbose:
            sys.stdout.write("\r{0} is calculating...".format(name))
            sys.stdout.flush()

        matchingCostFunc, backTrackFunc = check_lowmemory_myMatchingCostFunc(myMatchingCostFunc)

        # for check
        matchingCost = matchingCostFunc(refDatas, inpDatas)

        try:
            matchingCost = matchingCostFunc(refDatas, inpDatas)
        except:
            if self.verbose:
                sys.stdout.write("\rWarning:{0}:{1}\nskip...\n".format(name, sys.exc_info()))
                sys.stdout.flush()
                return None, None
        """
        for check
        
        correspondentPoints = backTrackFunc(matchingCost=matchingCost, refDatas=refDatas, inpDatas=inpDatas,
                                            argsFinFrameBackTracked=np.unravel_index(np.nanargmin(matchingCost[matchingCost.shape[0] - 1]),
                                             matchingCost[matchingCost.shape[0] - 1].shape))
        """
        try:
            correspondentPointses = backTrackFunc(matchingCost=matchingCost, refDatas=refDatas, inpDatas=inpDatas,
                                                  argsFinFrameBackTracked=np.unravel_index(np.nanargmin(matchingCost[matchingCost.shape[0] - 1]),
                                                      matchingCost[matchingCost.shape[0] - 1].shape))
        except ValueError:
            # if self.verbose:
            if self.verboseNan:
                sys.stdout.write("\rWarning:{0}\'s all matching cost has nan\nskip...\n".format(name))
                sys.stdout.flush()
            return None, matchingCost

        if self.verbose:
            sys.stdout.write("\r{0} is calculating...finished\n".format(name))
            sys.stdout.flush()
        return correspondentPointses, matchingCost

    def save(self, **kwargs):
        pass

def check_myMatchingCostFunc(myMatchingCostFunc):
    # return matchingCostFunc, backTrackFunc
    if myMatchingCostFunc is not None and not isinstance(myMatchingCostFunc, dict):
        raise ValueError('myMatchingCostFunc must be dict, and one\'s key must have [\'matchingCost\',\'backTrack\']')
    else:
        if myMatchingCostFunc is None:
            myMatchingCostFunc = constraint('default')
        try:
            return myMatchingCostFunc['matchingCost'], myMatchingCostFunc['backTrack']
        except KeyError:
            raise KeyError('myMatchingCostFunc must be dict, and one\'s key must have [\'matchingCost\',\'backTrack\']')

def check_args(refDatas, inpDatas):
    if len(refDatas) != len(inpDatas):
        raise KeyError('key\'s number must be even and more than 2, but got r:{0}, i:{1}'.format(len(refDatas), len(inpDatas)))

    num = int(len(refDatas) / 2)
    refShapes = np.array([refDatas[i].shape for i in range(num)])
    inpShapes = np.array([inpDatas[i].shape for i in range(num)])

    if np.unique(refShapes[:, 1]).size > 1 or np.unique(inpShapes[:, 1]).size > 1 \
            or np.sum(refShapes[:, 1] == inpShapes[:, 1]) != num:
        raise ValueError('refData and inpData must have same dimension')

    if np.unique(refShapes[:, 0]).size > 1:
        raise ValueError('refData must have same time')
    if np.unique(inpShapes[:, 0]).size > 1:
        raise ValueError('inpData must have same time')

    return

def check_lowmemory_myMatchingCostFunc(myMatchingCostFunc):
    # return matchingCostFunc, backTrackFunc
    if not isinstance(myMatchingCostFunc, dict):
        raise ValueError('myMatchingCostFunc must be dict, and one\'s key must have [\'matchingCost\',\'backTrack\']')
    else:
        try:
            return myMatchingCostFunc['matchingCost'], myMatchingCostFunc['backTrack']
        except KeyError:
            raise KeyError('myMatchingCostFunc must be dict, and one\'s key must have [\'matchingCost\',\'backTrack\']')
