import numpy as np
from scipy.spatial.distance import cdist

def constraint(kind='default'):
    if kind == 'default' or kind == 'asynm':
        def asynmCalc(localCost):
            matchingCost = np.zeros(localCost.shape)
            matchingCost[0, :] = localCost[0, :]

            referenceTimeMax = localCost.shape[0]
            for referenceTime in range(1, referenceTimeMax):
                matchingCost[referenceTime, 0] = localCost[referenceTime, 0] + matchingCost[referenceTime - 1, 0]
                matchingCost[referenceTime, 1] = localCost[referenceTime, 1] + np.minimum(
                    matchingCost[referenceTime - 1, 0],
                    matchingCost[referenceTime - 1, 1])
                matchingCost[referenceTime, 2:] = localCost[referenceTime, 2:] + np.minimum.reduce(
                    [matchingCost[referenceTime - 1, 2:],
                     matchingCost[referenceTime - 1, 1:-1],
                     matchingCost[referenceTime - 1, :-2]])

            return matchingCost

        def asynmBackTrack(**kwargs):
            matchingCost = kwargs['matchingCost']
            inputFinFrameBackTracked = kwargs['inputFinFrameBackTracked']
            # back track
            correspondentPoints = []
            r, i = matchingCost.shape[0] - 1, inputFinFrameBackTracked
            correspondentPoints.append([r, i])

            while r > 0 and i > 1:
                tmp = np.argmin((matchingCost[r - 1, i], matchingCost[r - 1, i - 1],
                                 matchingCost[r - 1, i - 2]))
                r = r - 1
                i = i - tmp
                correspondentPoints.insert(0, [r, i])

            while r > 0 and i > 0:
                tmp = np.argmin((matchingCost[r - 1, i], matchingCost[r - 1, i - 1]))
                r = r - 1
                i = i - tmp
                correspondentPoints.insert(0, [r, i])

            while r > 0:
                r = r - 1
                i = 0
                correspondentPoints.insert(0, [r, i])

            return correspondentPoints

        return {'matchingCost': asynmCalc, 'backTrack': asynmBackTrack}

    elif kind == 'visualization' or kind == 'sync':
        def syncCalc(localCost):
            matchingCost = np.zeros(localCost.shape)
            matchingCost[0, :] = localCost[0, :]
            # pointless part is infinity
            matchingCost[1:, 0] = np.inf
            # only m(r-1,i-1) + 2*l(r,i) part (only (1,1))
            matchingCost[1, 1] = matchingCost[0, 0] + 2*localCost[1, 1]
            # m(r-1,i-2) + 2*l(r,i-1) + l(r,i), m(r-1,i-1) + 2*l(r,i) part ((1,2:))
            matchingCost[1, 2:] = np.minimum(matchingCost[0, :-2] + 2*localCost[1, 1:-1] + localCost[1, 2:],
                                             matchingCost[0, 1:-1] + 2*localCost[1, 2:])
            # m(r-1,i-1) + 2*l(r,i), m(r-2,i-1) + 2*l(r-1,i) + l(r,i) part ((2:,1))
            matchingCost[2:, 1] = np.minimum(matchingCost[1:-1, 0] + 2*localCost[2:, 1],
                                             matchingCost[0:-2, 0] + 2*localCost[1:-1, 1] + localCost[2:, 1])
            # all pathes part (other)
            referenceTimeMax = localCost.shape[0]
            for referenceTime in range(2, referenceTimeMax):
                matchingCost[referenceTime, 2:] = np.minimum.reduce(
                    [matchingCost[referenceTime - 2, 1:-1] + 2*localCost[referenceTime - 1, 2:] + localCost[referenceTime, 2:],
                     matchingCost[referenceTime - 1, 1:-1] + 2*localCost[referenceTime, 2:],
                     matchingCost[referenceTime - 1, :-2] + 2*localCost[referenceTime, 1:-1] + localCost[referenceTime, 2:]])
            if np.sum(np.isinf(matchingCost[referenceTimeMax - 1, :])) == matchingCost.shape[1]: # all matching cost are infinity
                raise OverflowError('all matching cost are infinity')
            return matchingCost
        if kind == 'sync':
            def syncBackTrack(**kwargs):
                localCost = kwargs['localCost']
                matchingCost = kwargs['matchingCost']
                inputFinFrameBackTracked = kwargs['inputFinFrameBackTracked']

                correspondentPoints = []
                r, i = matchingCost.shape[0] - 1, inputFinFrameBackTracked
                correspondentPoints.append([r, i])

                while r > 1 and i > 1:
                    tmp = np.argmin((matchingCost[r - 2, i - 1] + 2 * localCost[r - 1, i],
                                     matchingCost[r - 1, i - 1] + localCost[r, i],
                                     matchingCost[r - 1, i - 2] + 2 * localCost[r, i - 1]))

                    if tmp == 0:
                        correspondentPoints.insert(0, [r - 1, i])
                        r = r - 2
                        i = i - 1
                        correspondentPoints.insert(0, [r, i])
                    elif tmp == 1:
                        r = r - 1
                        i = i - 1
                        correspondentPoints.insert(0, [r, i])
                    else:
                        correspondentPoints.insert(0, [r, i - 1])
                        r = r - 1
                        i = i - 2
                        correspondentPoints.insert(0, [r, i])

                if r == 2 and i == 1:
                    correspondentPoints.insert(0, [r - 1, i])
                    r = r - 2
                    i = i - 1
                    correspondentPoints.insert(0, [r, i])
                elif r == 1 and i == 1:
                    correspondentPoints.insert(0, [r - 1, i - 1])
                elif r == 1 and i > 1:
                    tmp = np.argmin((matchingCost[r - 1, i - 1] + localCost[r, i],
                                     matchingCost[r - 1, i - 2] + 2 * localCost[r, i - 1]))

                    if tmp == 0:
                        r = r - 1
                        i = i - 1
                        correspondentPoints.insert(0, [r, i])
                    elif tmp == 1:
                        correspondentPoints.insert(0, [r, i - 1])
                        r = r - 1
                        i = i - 2
                        correspondentPoints.insert(0, [r, i])

                return correspondentPoints
        else: # visualization, all input time are matched
            def syncBackTrack(**kwargs):
                localCost = kwargs['localCost']
                matchingCost = kwargs['matchingCost']
                inputFinFrameBackTracked = kwargs['inputFinFrameBackTracked']

                correspondentPoints = []
                r, i = matchingCost.shape[0] - 1, inputFinFrameBackTracked
                correspondentPoints.append([r, i])

                while r > 1 and i > 1:
                    tmp = np.argmin((matchingCost[r - 2, i - 1] + 2 * localCost[r - 1, i],
                                     matchingCost[r - 1, i - 1] + localCost[r, i],
                                     matchingCost[r - 1, i - 2] + 2 * localCost[r, i - 1]))

                    if tmp == 0: # slope = 2 (d(inp)/d(ref))
                        r = r - 2
                        i = i - 1
                        correspondentPoints.insert(0, [r, i])
                    elif tmp == 1: # slope = 1 (d(inp)/d(ref))
                        r = r - 1
                        i = i - 1
                        correspondentPoints.insert(0, [r, i])
                    else: # slope = 1/2 (d(inp)/d(ref))
                        correspondentPoints.insert(0, [r, i - 1])
                        r = r - 1
                        i = i - 2
                        correspondentPoints.insert(0, [r, i])


                if r == 2:
                    r = r - 2
                    i = i - 1
                    correspondentPoints.insert(0, [r, i])
                elif r == 1 and i == 1:
                    correspondentPoints.insert(0, [r - 1, i - 1])
                elif r == 1 and i > 1:
                    tmp = np.argmin((matchingCost[r - 1, i - 1] + localCost[r, i],
                                     matchingCost[r - 1, i - 2] + 2 * localCost[r, i - 1]))

                    if tmp == 0:
                        r = r - 1
                        i = i - 1
                        correspondentPoints.insert(0, [r, i])
                    elif tmp == 1:
                        correspondentPoints.insert(0, [r, i - 1])
                        r = r - 1
                        i = i - 2
                        correspondentPoints.insert(0, [r, i])

                return correspondentPoints

        return {'matchingCost': syncCalc, 'backTrack': syncBackTrack}
        """
        # for debug
            a = np.array([[1,5,6,2], [1,3,3,3], [3,2,1,0], [3,4,2,2], [2,3,1,3]])
            print('\n')
            print(a)
            print(a.shape)
            matchingCost = np.zeros(a.shape)
            matchingCost[0, :] = a[0, :]
            # pointless part is infinity
            matchingCost[1:, 0] = np.inf
            # only m(r-1,i-1) + 2*l(r,i) part (only (1,1))
            matchingCost[1, 1] = matchingCost[0, 0] + 2 * a[1, 1]
            # m(r-1,i-2) + 2*l(r,i-1) + l(r,i), m(r-1,i-1) + 2*l(r,i) part ((1,2:))
            matchingCost[1, 2:] = np.minimum(matchingCost[0, :-2] + 2 * a[1, 1:-1] + a[1, 2:],
                                             matchingCost[0, 1:-1] + 2 * a[1, 2:])
            # m(r-1,i-1) + 2*l(r,i), m(r-2,i-1) + 2*l(r-1,i) + l(r,i) part ((2:,1))
            matchingCost[2:, 1] = np.minimum(matchingCost[1:-1, 0] + 2 * a[2:, 1],
                                             matchingCost[0:-2, 0] + 2 * a[1:-1, 1] + a[2:, 1])
            # all pathes part (other)
            referenceTimeMax = a.shape[0]
            for referenceTime in range(2, referenceTimeMax):
                matchingCost[referenceTime, 2:] = np.minimum.reduce(
                    [matchingCost[referenceTime - 2, 1:-1] + 2 * a[referenceTime - 1, 2:] + a[
                                                                                                     referenceTime, 2:],
                     matchingCost[referenceTime - 1, 1:-1] + 2 * a[referenceTime, 2:],
                     matchingCost[referenceTime - 1, :-2] + 2 * a[referenceTime, 1:-1] + a[
                                                                                                  referenceTime, 2:]])
            print(matchingCost)

            correspondentPoints = []
            r, i = matchingCost.shape[0] - 1, np.argmin(matchingCost[referenceTimeMax - 1, :])
            correspondentPoints.append([r, i])

            while r > 1 and i > 1:
                tmp = np.argmin((matchingCost[r - 2, i - 1] + 2 * a[r - 1, i],
                                 matchingCost[r - 1, i - 1] + a[r, i],
                                 matchingCost[r - 1, i - 2] + 2 * a[r, i - 1]))

                if tmp == 0:
                    correspondentPoints.insert(0, [r - 1, i])
                    r = r - 2
                    i = i - 1
                    correspondentPoints.insert(0, [r, i])
                elif tmp == 1:
                    r = r - 1
                    i = i - 1
                    correspondentPoints.insert(0, [r, i])
                else:
                    correspondentPoints.insert(0, [r, i - 1])
                    r = r - 1
                    i = i - 2
                    correspondentPoints.insert(0, [r, i])

            if r == 2 and i == 1:
                correspondentPoints.insert(0, [r - 1, i])
                r = r - 2
                i = i - 1
                correspondentPoints.insert(0, [r, i])
            elif r == 1 and i == 1:
                correspondentPoints.insert(0, [r - 1, i - 1])
            else:  # r > 0 and i == 1
                tmp = np.argmin((matchingCost[r - 1, i - 1] + a[r, i],
                                 matchingCost[r - 1, i - 2] + 2 * a[r, i - 1]))

                if tmp == 0:
                    r = r - 1
                    i = i - 1
                    correspondentPoints.insert(0, [r, i])
                elif tmp == 1:
                    correspondentPoints.insert(0, [r, i - 1])
                    r = r - 1
                    i = i - 2
                    correspondentPoints.insert(0, [r, i])
            print(correspondentPoints)
            exit()
        """

    elif kind == 'async2': # greedy method
        pass
    else:
        raise ValueError('{0} is invalid constraint name'.format(kind))


def lowMemoryConstraint(kind='default'):
    if kind == 'default' or kind == 'asynm':
        def asynmCalc(localCost):
            matchingCost = np.zeros(localCost.shape)
            matchingCost[0, :] = localCost[0, :]

            referenceTimeMax = localCost.shape[0]
            for referenceTime in range(1, referenceTimeMax):
                matchingCost[referenceTime, 0] = localCost[referenceTime, 0] + matchingCost[referenceTime - 1, 0]
                matchingCost[referenceTime, 1] = localCost[referenceTime, 1] + np.minimum(
                    matchingCost[referenceTime - 1, 0],
                    matchingCost[referenceTime - 1, 1])
                matchingCost[referenceTime, 2:] = localCost[referenceTime, 2:] + np.minimum.reduce(
                    [matchingCost[referenceTime - 1, 2:],
                     matchingCost[referenceTime - 1, 1:-1],
                     matchingCost[referenceTime - 1, :-2]])

            return matchingCost

        def asynmBackTrack(**kwargs):
            matchingCost = kwargs['matchingCost']
            inputFinFrameBackTracked = kwargs['inputFinFrameBackTracked']
            # back track
            correspondentPoints = []
            r, i = matchingCost.shape[0] - 1, inputFinFrameBackTracked
            correspondentPoints.append([r, i])

            while r > 0 and i > 1:
                tmp = np.argmin((matchingCost[r - 1, i], matchingCost[r - 1, i - 1],
                                 matchingCost[r - 1, i - 2]))
                r = r - 1
                i = i - tmp
                correspondentPoints.insert(0, [r, i])

            while r > 0 and i > 0:
                tmp = np.argmin((matchingCost[r - 1, i], matchingCost[r - 1, i - 1]))
                r = r - 1
                i = i - tmp
                correspondentPoints.insert(0, [r, i])

            while r > 0:
                r = r - 1
                i = 0
                correspondentPoints.insert(0, [r, i])

            return correspondentPoints

        return {'matchingCost': asynmCalc, 'backTrack': asynmBackTrack}

    elif kind == 'visualization' or kind == 'sync':
        def syncCalc(localCost):
            matchingCost = np.zeros(localCost.shape)
            matchingCost[0, :] = localCost[0, :]
            # pointless part is infinity
            matchingCost[1:, 0] = np.inf
            # only m(r-1,i-1) + 2*l(r,i) part (only (1,1))
            matchingCost[1, 1] = matchingCost[0, 0] + 2*localCost[1, 1]
            # m(r-1,i-2) + 2*l(r,i-1) + l(r,i), m(r-1,i-1) + 2*l(r,i) part ((1,2:))
            matchingCost[1, 2:] = np.minimum(matchingCost[0, :-2] + 2*localCost[1, 1:-1] + localCost[1, 2:],
                                             matchingCost[0, 1:-1] + 2*localCost[1, 2:])
            # m(r-1,i-1) + 2*l(r,i), m(r-2,i-1) + 2*l(r-1,i) + l(r,i) part ((2:,1))
            matchingCost[2:, 1] = np.minimum(matchingCost[1:-1, 0] + 2*localCost[2:, 1],
                                             matchingCost[0:-2, 0] + 2*localCost[1:-1, 1] + localCost[2:, 1])
            # all pathes part (other)
            referenceTimeMax = localCost.shape[0]
            for referenceTime in range(2, referenceTimeMax):
                matchingCost[referenceTime, 2:] = np.minimum.reduce(
                    [matchingCost[referenceTime - 2, 1:-1] + 2*localCost[referenceTime - 1, 2:] + localCost[referenceTime, 2:],
                     matchingCost[referenceTime - 1, 1:-1] + 2*localCost[referenceTime, 2:],
                     matchingCost[referenceTime - 1, :-2] + 2*localCost[referenceTime, 1:-1] + localCost[referenceTime, 2:]])
            if np.sum(np.isinf(matchingCost[referenceTimeMax - 1, :])) == matchingCost.shape[1]: # all matching cost are infinity
                raise OverflowError('all matching cost are infinity')
            return matchingCost
        if kind == 'sync':
            def syncBackTrack(**kwargs):
                localCost = kwargs['localCost']
                matchingCost = kwargs['matchingCost']
                inputFinFrameBackTracked = kwargs['inputFinFrameBackTracked']

                correspondentPoints = []
                r, i = matchingCost.shape[0] - 1, inputFinFrameBackTracked
                correspondentPoints.append([r, i])

                while r > 1 and i > 1:
                    tmp = np.argmin((matchingCost[r - 2, i - 1] + 2 * localCost[r - 1, i],
                                     matchingCost[r - 1, i - 1] + localCost[r, i],
                                     matchingCost[r - 1, i - 2] + 2 * localCost[r, i - 1]))

                    if tmp == 0:
                        correspondentPoints.insert(0, [r - 1, i])
                        r = r - 2
                        i = i - 1
                        correspondentPoints.insert(0, [r, i])
                    elif tmp == 1:
                        r = r - 1
                        i = i - 1
                        correspondentPoints.insert(0, [r, i])
                    else:
                        correspondentPoints.insert(0, [r, i - 1])
                        r = r - 1
                        i = i - 2
                        correspondentPoints.insert(0, [r, i])

                if r == 2 and i == 1:
                    correspondentPoints.insert(0, [r - 1, i])
                    r = r - 2
                    i = i - 1
                    correspondentPoints.insert(0, [r, i])
                elif r == 1 and i == 1:
                    correspondentPoints.insert(0, [r - 1, i - 1])
                elif r == 1 and i > 1:
                    tmp = np.argmin((matchingCost[r - 1, i - 1] + localCost[r, i],
                                     matchingCost[r - 1, i - 2] + 2 * localCost[r, i - 1]))

                    if tmp == 0:
                        r = r - 1
                        i = i - 1
                        correspondentPoints.insert(0, [r, i])
                    elif tmp == 1:
                        correspondentPoints.insert(0, [r, i - 1])
                        r = r - 1
                        i = i - 2
                        correspondentPoints.insert(0, [r, i])

                return correspondentPoints
        else: # visualization, all input time are matched
            def syncBackTrack(**kwargs):
                localCost = kwargs['localCost']
                matchingCost = kwargs['matchingCost']
                inputFinFrameBackTracked = kwargs['inputFinFrameBackTracked']

                correspondentPoints = []
                r, i = matchingCost.shape[0] - 1, inputFinFrameBackTracked
                correspondentPoints.append([r, i])

                while r > 1 and i > 1:
                    tmp = np.argmin((matchingCost[r - 2, i - 1] + 2 * localCost[r - 1, i],
                                     matchingCost[r - 1, i - 1] + localCost[r, i],
                                     matchingCost[r - 1, i - 2] + 2 * localCost[r, i - 1]))

                    if tmp == 0: # slope = 2 (d(inp)/d(ref))
                        r = r - 2
                        i = i - 1
                        correspondentPoints.insert(0, [r, i])
                    elif tmp == 1: # slope = 1 (d(inp)/d(ref))
                        r = r - 1
                        i = i - 1
                        correspondentPoints.insert(0, [r, i])
                    else: # slope = 1/2 (d(inp)/d(ref))
                        correspondentPoints.insert(0, [r, i - 1])
                        r = r - 1
                        i = i - 2
                        correspondentPoints.insert(0, [r, i])


                if r == 2:
                    r = r - 2
                    i = i - 1
                    correspondentPoints.insert(0, [r, i])
                elif r == 1 and i == 1:
                    correspondentPoints.insert(0, [r - 1, i - 1])
                elif r == 1 and i > 1:
                    tmp = np.argmin((matchingCost[r - 1, i - 1] + localCost[r, i],
                                     matchingCost[r - 1, i - 2] + 2 * localCost[r, i - 1]))

                    if tmp == 0:
                        r = r - 1
                        i = i - 1
                        correspondentPoints.insert(0, [r, i])
                    elif tmp == 1:
                        correspondentPoints.insert(0, [r, i - 1])
                        r = r - 1
                        i = i - 2
                        correspondentPoints.insert(0, [r, i])

                return correspondentPoints

        return {'matchingCost': syncCalc, 'backTrack': syncBackTrack}

    elif kind == 'async2':
        limits = 2
        def _localCost(refData1, refData2, inpData1, inpData2):
            R, I = refData1.shape[0], inpData1.shape[0]

            localCosts = np.zeros((R, I, 2*limits + 1)) # (r, i, epsilon)
            # minus epsilon
            for epsilon in range(-limits, 0):
                index = epsilon + limits
                # d_{i,j} + d'_{i,j+epsilon}
                # d_{i,j}
                localCosts[:, :, index] += cdist(refData1, inpData1, 'euclidean')
                # d'_{i,j+epsilon}
                localCosts[:, -epsilon:, index] += cdist(refData2, inpData2[:epsilon], 'euclidean')
                # if j+epsilon is less than zero, then inf
                localCosts[:, :-epsilon, index] = np.inf

            # epsilon = 0
            localCosts[:, :, limits] += cdist(refData1, inpData1, 'euclidean')
            # d'_{i,j+epsilon}
            localCosts[:, :, limits] += cdist(refData2, inpData2, 'euclidean')

            # plus epsilon
            for epsilon in range(1, limits + 1):
                index = epsilon + limits
                # d_{i,j} + d'_{i,j+epsilon}
                # d_{i,j}
                localCosts[:, :, index] += cdist(refData1, inpData1, 'euclidean')
                # d'_{i,j+epsilon}
                localCosts[:, :-epsilon, index] += cdist(refData2, inpData2[epsilon:], 'euclidean')
                # if j+epsilon is less than zero, then inf
                localCosts[:, -epsilon:, index] = np.inf

            return localCosts


        def asyncCalc(refDatas, inpDatas):
            if len(refDatas) != 2 or len(inpDatas) != 2:
                raise ValueError('The length of both refDatas and inpDatas must be two, but got ref:{0} and inp:{1}'
                                 .format(len(refDatas), len(inpDatas)))
            R, I = refDatas[0].shape[0], inpDatas[0].shape[0]

            localCost = _localCost(refDatas[0], refDatas[1], inpDatas[0], inpDatas[1])
            matchingCost = np.zeros((R, I, 2*limits + 1))
            matchingCost[0, :, :] = localCost[0, :, :]
            #use slice instead of constraintInds = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
            for r in range(1, R):
                # i = 0
                for index, epsilon in enumerate(range(-limits, limits + 1)):
                    matchingCost[r, 0, index] = localCost[r, 0, index] + np.min(matchingCost[r - 1, 0, max(epsilon, 0):index + 1])
                # i = 1
                for index, epsilon in enumerate(range(-limits, limits + 1)):
                    matchingCost[r, 1, index] = localCost[r, 1, index] + np.min(np.concatenate([matchingCost[r - 1, 1, max(epsilon, 0):index + 1],
                                                                                                matchingCost[r - 1, 0, max(epsilon + 1, 0):index + 2]]))

                # i = 2...
                for index, epsilon in enumerate(range(-limits, limits + 1)):
                    matchingCost[r, 2:, index] = localCost[r, 2:, index] + np.minimum.reduce(np.concatenate(
                                                                                            [matchingCost[r - 1, 2:, max(epsilon, 0):index + 1],
                                                                                            matchingCost[r - 1, 1:-1, max(epsilon + 1, 0):index + 2],
                                                                                            matchingCost[r - 1, :-2, max(epsilon + 2, 0):index + 3]], axis=1), axis=1)

            if np.sum(np.isinf(matchingCost[R - 1, :, :])) == matchingCost.shape[1] * matchingCost.shape[2]: # all matching cost are infinity
                raise OverflowError('all matching cost are infinity')
            return matchingCost

        def asyncBackTrack(**kwargs):
            refDatas = kwargs['refDatas']
            inpDatas = kwargs['inpDatas']
            matchingCost = kwargs['matchingCost']
            inputFinFrameBackTracked, epsilonFinFrameBackTracked = kwargs['argsFinFrameBackTracked']

            correspondentPoints1, correspondentPoints2 = [], []
            r, i, epsilon = matchingCost.shape[0] - 1, inputFinFrameBackTracked, epsilonFinFrameBackTracked - limits
            correspondentPoints1.append([r, i])
            correspondentPoints2.append([r, i + epsilon])

            """
            epsilon\i| i     i-1   i-2(tmp=0,1,2)
            -------------------------
                -2    | e+c   e+c   e+c
                -1    | e+c-1 e+c-1 e+c
                0     | e+c-2 e+c-1 e+c
                1     | e+c-2 e+c-1 e+c
                2     | e+c-2 e+c-1 e+c
            (epsilon + limits=0,...,4)
            """

            forNewEpsilon = [[0, 0, 0],
                             [-1, -1, 0],
                             [-2, -1, 0],
                             [-2, -1, 0],
                             [-2, -1, 0]]

            while r > 0:
                if i > 1:
                    mcosts = [matchingCost[r - 1, i, max(epsilon, 0):epsilon + limits + 1],
                              matchingCost[r - 1, i - 1, max(epsilon + 1, 0):epsilon + limits + 2],
                              matchingCost[r - 1, i - 2, max(epsilon + 2, 0):epsilon + limits + 3]]
                    c_, tmp = [], []
                    for ind, mcost in enumerate(mcosts):
                        c_.append(np.argmin(mcost))
                        tmp.append(mcost[c_[ind]])

                    tmp = np.argmin(tmp)

                    r = r - 1
                    i = i - tmp
                    epsilon = epsilon + c_[tmp] + forNewEpsilon[epsilon + limits][tmp]
                    correspondentPoints1.insert(0, [r, i])
                    correspondentPoints2.insert(0, [r, i + epsilon])
                    """
                    # i -> i
                    if tmp == 0:
                        r = r - 1
                        epsilon = epsilon + c_[tmp] + forNewEpsilon[epsilon + limits][tmp]
                        correspondentPoints1.insert(0, [r, i])
                        correspondentPoints2.insert(0, [r, i + epsilon])

                    # i -> i - 1
                    elif tmp == 1:
                        r = r - 1
                        i = i - 1
                        epsilon = epsilon + c_[tmp] + forNewEpsilon[epsilon + limits][tmp]
                        correspondentPoints1.insert(0, [r, i])
                        correspondentPoints2.insert(0, [r, i + epsilon])
                    else:  # i -> i - 2
                        r = r - 1
                        i = i - 2
                        epsilon = epsilon + c_[tmp] + forNewEpsilon[epsilon + limits][tmp]
                        correspondentPoints1.insert(0, [r, i])
                        correspondentPoints2.insert(0, [r, i + epsilon])
                    """
                elif i == 1:
                    mcosts = [matchingCost[r - 1, 1, max(epsilon, 0):epsilon + limits + 1],
                              matchingCost[r - 1, 0, max(epsilon + 1, 0):epsilon + limits + 2]]

                    c_, tmp = [], []
                    for ind, mcost in enumerate(mcosts):
                        c_.append(np.argmin(mcost))
                        tmp.append(mcost[c_[ind]])

                    tmp = np.argmin(tmp)

                    r = r - 1
                    i = i - tmp
                    epsilon = epsilon + c_[tmp] + forNewEpsilon[epsilon + limits][tmp]
                    correspondentPoints1.insert(0, [r, i])
                    correspondentPoints2.insert(0, [r, i + epsilon])

                else: # i == 0
                    c = np.argmin(matchingCost[r - 1, 0, max(epsilon, 0):epsilon + limits + 1])

                    r = r - 1
                    #i = i
                    epsilon = epsilon + c + forNewEpsilon[epsilon + limits][0]
                    correspondentPoints1.insert(0, [r, i])
                    correspondentPoints2.insert(0, [r, i + epsilon])

            return correspondentPoints1, correspondentPoints2

        return {'matchingCost': asyncCalc, 'backTrack': asyncBackTrack}

    elif kind == 'async3-lined':
        limits = 2
        def _localCost(refData1, refData2, refData3, inpData1, inpData2, inpData3):
            R, I = refData1.shape[0], inpData1.shape[0]

            localCosts = np.zeros((R, I, 2*limits + 1)) # (r, i, epsilon)
            # minus epsilon
            for epsilon in range(-limits, 0):
                index = epsilon + limits
                # d_{i,j} + d'_{i,j+epsilon}
                # d_{i,j}
                localCosts[:, :, index] += cdist(refData1, inpData1, 'euclidean')
                # d'_{i,j+epsilon}
                localCosts[:, -epsilon:, index] += cdist(refData2, inpData2[:epsilon], 'euclidean')
                # if j+epsilon is less than zero, then inf
                localCosts[:, :-epsilon, index] = np.inf

            # epsilon = 0
            localCosts[:, :, limits] += cdist(refData1, inpData1, 'euclidean')
            # d'_{i,j+epsilon}
            localCosts[:, :, limits] += cdist(refData2, inpData2, 'euclidean')

            # plus epsilon
            for epsilon in range(1, limits + 1):
                index = epsilon + limits
                # d_{i,j} + d'_{i,j+epsilon}
                # d_{i,j}
                localCosts[:, :, index] += cdist(refData1, inpData1, 'euclidean')
                # d'_{i,j+epsilon}
                localCosts[:, :-epsilon, index] += cdist(refData2, inpData2[epsilon:], 'euclidean')
                # if j+epsilon is less than zero, then inf
                localCosts[:, -epsilon:, index] = np.inf

            return localCosts

        def asyncCalc(refDatas, inpDatas):
            R, I = refDatas[0].shape[0], inpDatas[0].shape[0]

            localCost = _localCost(refDatas[0], refDatas[1], inpDatas[0], inpDatas[1])
            matchingCost = np.zeros((R, I, 2*limits + 1))
            matchingCost[0, :, :] = localCost[0, :, :]
            #use slice instead of constraintInds = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
            for r in range(1, R):
                # i = 0
                for index, epsilon in enumerate(range(-limits, limits + 1)):
                    matchingCost[r, 0, index] = localCost[r, 0, index] + np.min(matchingCost[r - 1, 0, max(epsilon, 0):index + 1])
                # i = 1
                for index, epsilon in enumerate(range(-limits, limits + 1)):
                    matchingCost[r, 1, index] = localCost[r, 1, index] + np.min(np.concatenate([matchingCost[r - 1, 1, max(epsilon, 0):index + 1],
                                                                                                matchingCost[r - 1, 0, max(epsilon + 1, 0):index + 2]]))

                # i = 2...
                for index, epsilon in enumerate(range(-limits, limits + 1)):
                    matchingCost[r, 2:, index] = localCost[r, 2:, index] + np.minimum.reduce(np.concatenate(
                                                                                            [matchingCost[r - 1, 2:, max(epsilon, 0):index + 1],
                                                                                            matchingCost[r - 1, 1:-1, max(epsilon + 1, 0):index + 2],
                                                                                            matchingCost[r - 1, :-2, max(epsilon + 2, 0):index + 3]], axis=1), axis=1)

            if np.sum(np.isinf(matchingCost[R - 1, :, :])) == matchingCost.shape[1] * matchingCost.shape[2]: # all matching cost are infinity
                raise OverflowError('all matching cost are infinity')
            return matchingCost

        def asyncBackTrack(**kwargs):
            refDatas = kwargs['refDatas']
            inpDatas = kwargs['inpDatas']
            matchingCost = kwargs['matchingCost']
            inputFinFrameBackTracked, epsilonFinFrameBackTracked = kwargs['argsFinFrameBackTracked']

            correspondentPoints1, correspondentPoints2 = [], []
            r, i, epsilon = matchingCost.shape[0] - 1, inputFinFrameBackTracked, epsilonFinFrameBackTracked - limits
            correspondentPoints1.append([r, i])
            correspondentPoints2.append([r, i + epsilon])

            """
            epsilon\i| i     i-1   i-2(tmp=0,1,2)
            -------------------------
                -2    | e+c   e+c   e+c
                -1    | e+c-1 e+c-1 e+c
                0     | e+c-2 e+c-1 e+c
                1     | e+c-2 e+c-1 e+c
                2     | e+c-2 e+c-1 e+c
            (epsilon + limits=0,...,4)
            """

            forNewEpsilon = [[0, 0, 0],
                             [-1, -1, 0],
                             [-2, -1, 0],
                             [-2, -1, 0],
                             [-2, -1, 0]]

            while r > 0:
                if i > 1:
                    mcosts = [matchingCost[r - 1, i, max(epsilon, 0):epsilon + limits + 1],
                              matchingCost[r - 1, i - 1, max(epsilon + 1, 0):epsilon + limits + 2],
                              matchingCost[r - 1, i - 2, max(epsilon + 2, 0):epsilon + limits + 3]]
                    c_, tmp = [], []
                    for ind, mcost in enumerate(mcosts):
                        c_.append(np.argmin(mcost))
                        tmp.append(mcost[c_[ind]])

                    tmp = np.argmin(tmp)

                    r = r - 1
                    i = i - tmp
                    epsilon = epsilon + c_[tmp] + forNewEpsilon[epsilon + limits][tmp]
                    correspondentPoints1.insert(0, [r, i])
                    correspondentPoints2.insert(0, [r, i + epsilon])
                    """
                    # i -> i
                    if tmp == 0:
                        r = r - 1
                        epsilon = epsilon + c_[tmp] + forNewEpsilon[epsilon + limits][tmp]
                        correspondentPoints1.insert(0, [r, i])
                        correspondentPoints2.insert(0, [r, i + epsilon])

                    # i -> i - 1
                    elif tmp == 1:
                        r = r - 1
                        i = i - 1
                        epsilon = epsilon + c_[tmp] + forNewEpsilon[epsilon + limits][tmp]
                        correspondentPoints1.insert(0, [r, i])
                        correspondentPoints2.insert(0, [r, i + epsilon])
                    else:  # i -> i - 2
                        r = r - 1
                        i = i - 2
                        epsilon = epsilon + c_[tmp] + forNewEpsilon[epsilon + limits][tmp]
                        correspondentPoints1.insert(0, [r, i])
                        correspondentPoints2.insert(0, [r, i + epsilon])
                    """
                elif i == 1:
                    mcosts = [matchingCost[r - 1, 1, max(epsilon, 0):epsilon + limits + 1],
                              matchingCost[r - 1, 0, max(epsilon + 1, 0):epsilon + limits + 2]]

                    c_, tmp = [], []
                    for ind, mcost in enumerate(mcosts):
                        c_.append(np.argmin(mcost))
                        tmp.append(mcost[c_[ind]])

                    tmp = np.argmin(tmp)

                    r = r - 1
                    i = i - tmp
                    epsilon = epsilon + c_[tmp] + forNewEpsilon[epsilon + limits][tmp]
                    correspondentPoints1.insert(0, [r, i])
                    correspondentPoints2.insert(0, [r, i + epsilon])

                else: # i == 0
                    c = np.argmin(matchingCost[r - 1, 0, max(epsilon, 0):epsilon + limits + 1])

                    r = r - 1
                    #i = i
                    epsilon = epsilon + c + forNewEpsilon[epsilon + limits][0]
                    correspondentPoints1.insert(0, [r, i])
                    correspondentPoints2.insert(0, [r, i + epsilon])

            return correspondentPoints1, correspondentPoints2

        return {'matchingCost': asyncCalc, 'backTrack': asyncBackTrack}

    else:
        raise ValueError('{0} is invalid constraint name'.format(kind))