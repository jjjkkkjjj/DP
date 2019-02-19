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
            matchingCost = kwargs.pop('matchingCost')
            inputFinFrameBackTracked = kwargs.get('inputFinFrameBackTracked', np.nanargmin(matchingCost[matchingCost.shape[0] - 1]))
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
                localCost = kwargs.pop('localCost')
                matchingCost = kwargs.pop('matchingCost')
                inputFinFrameBackTracked = kwargs.get('inputFinFrameBackTracked',
                                                      np.nanargmin(matchingCost[matchingCost.shape[0] - 1]))

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

    elif kind == 'visualization2' or kind == 'localdiff':
        def refskipCalc(localCost):
            matchingCost = np.zeros(localCost.shape)
            matchingCost[0, :] = localCost[0, :]
            # pointless part is infinity
            matchingCost[1:, 0] = np.inf
            matchingCost[1::2, :] = np.inf

            # all pathes part (other)
            referenceTimeMax = localCost.shape[0]
            for referenceTime in range(2, referenceTimeMax, 2):
                matchingCost[referenceTime, 1] = localCost[referenceTime, 1] + matchingCost[referenceTime - 2, 0]
                matchingCost[referenceTime, 2] = localCost[referenceTime, 2] + np.minimum(
                    matchingCost[referenceTime - 2, 0], matchingCost[referenceTime - 2, 1])
                matchingCost[referenceTime, 3] = localCost[referenceTime, 3] + np.min(
                    [matchingCost[referenceTime - 2, 0],
                     matchingCost[referenceTime - 2, 1],
                     matchingCost[referenceTime - 2, 2]])
                matchingCost[referenceTime, 4:] = localCost[referenceTime, 4:] + np.minimum.reduce(
                    [matchingCost[referenceTime - 2, 3:-1],
                     matchingCost[referenceTime - 2, 2:-2],
                     matchingCost[referenceTime - 2, 1:-3],
                     matchingCost[referenceTime - 2, :-4]])
            if referenceTimeMax % 2 == 0:
                matchingCost[referenceTimeMax - 1, :] = matchingCost[referenceTimeMax - 2, :]
            if np.sum(np.isinf(matchingCost[referenceTimeMax - 1, :])) == matchingCost.shape[1]: # all matching cost are infinity
                raise OverflowError('all matching cost are infinity')
            return matchingCost

        if kind == 'localdiff':
            def syncBackTrack(**kwargs):
                localCost = kwargs.get('localCost')
                matchingCost = kwargs.pop('matchingCost')
                inputFinFrameBackTracked = kwargs.get('inputFinFrameBackTracked',
                                                      np.nanargmin(matchingCost[matchingCost.shape[0] - 1]))

                correspondentPoints = []
                r, i = matchingCost.shape[0] - 1, inputFinFrameBackTracked

                if matchingCost.shape[0] % 2 == 0:
                    r = r - 1
                    correspondentPoints.insert(0, [r - 1, i])
                else:
                    correspondentPoints.insert(0, [r, i])

                while r > 0:
                    tmp = matchingCost[r - 2, max(0, i - 4):i - 1]  # i-4~i-1
                    tmp = np.argmin(tmp[::-1]) + 1  # value substracting i
                    # correspondentPoints.insert(0, [r - 1, i - tmp/2])
                    r = r - 2
                    i = i - tmp
                    correspondentPoints.insert(0, [r, i])

                return correspondentPoints

        else:
            def syncBackTrack(**kwargs):
                localCost = kwargs.get('localCost')
                matchingCost = kwargs.pop('matchingCost')
                inputFinFrameBackTracked = kwargs.get('inputFinFrameBackTracked', np.nanargmin(matchingCost[matchingCost.shape[0] - 1]))

                correspondentPoints = []
                r, i = matchingCost.shape[0] - 1, inputFinFrameBackTracked

                if matchingCost.shape[0] % 2 == 0:
                    r = r - 1
                    correspondentPoints.insert(0, [r - 1, i])
                else:
                    correspondentPoints.append([r, i])

                while r > 0:
                    tmp = matchingCost[r - 2, max(0, i - 4):i - 1]  # i-4~i-1
                    tmp = np.argmin(tmp[::-1]) + 1  # value substracting i
                    for tmp_i in range(1, tmp):
                        correspondentPoints.insert(0, [r - tmp_i*2/tmp, i - tmp_i])
                    # correspondentPoints.insert(0, [r - 1, i - tmp/2])
                    r = r - 2
                    i = i - tmp
                    correspondentPoints.insert(0, [r, i])

                return correspondentPoints

        return {'matchingCost': refskipCalc, 'backTrack': syncBackTrack}

    elif kind == 'async2': # greedy method
        pass
    else:
        raise NameError('{0} is invalid constraint name'.format(kind))


def lowMemoryConstraint(kind='default'):
    if kind == 'async2-asynm':
        limits = 2
        def _async2LocalCost(refDataBase, refDataPeripheral, inpDataBase, inpDataPeripheral):
            R, I = refDataBase.shape[0], inpDataBase.shape[0]

            localCosts = np.zeros((R, I, 2*limits + 1)) # (r, i, epsilon)
            for index, epsilon in enumerate(range(-limits, limits + 1)):
                # d_{i,j} + d'_{i,j+epsilon}
                # d_{i,j}
                localCosts[:, :, index] += cdist(refDataBase, inpDataBase, 'euclidean')
                # d'_{i,j+epsilon}
                localCosts[:, -min(0, epsilon):I + min(0, -epsilon), index] +=\
                    cdist(refDataPeripheral, inpDataPeripheral[max(0, epsilon):I - max(0, -epsilon)], 'euclidean')
                # if j+epsilon is no exist, then inf
                if epsilon < 0:
                    localCosts[:, :-epsilon, index] = np.inf
                elif epsilon > 0:
                    localCosts[:, -epsilon:, index] = np.inf
            """
            # minus epsilon
            for epsilon in range(-limits, 0):
                index = epsilon + limits
                # d_{i,j} + d'_{i,j+epsilon}
                # d_{i,j}
                localCosts[:, :, index] += cdist(refDataBase, inpDataBase, 'euclidean')
                # d'_{i,j+epsilon}
                localCosts[:, -epsilon:, index] += cdist(refDataPeripheral, inpDataPeripheral[:epsilon], 'euclidean')
                # if j+epsilon is less than zero, then inf
                localCosts[:, :-epsilon, index] = np.inf

            # epsilon = 0
            localCosts[:, :, limits] += cdist(refDataBase, inpDataBase, 'euclidean')
            # d'_{i,j+epsilon}
            localCosts[:, :, limits] += cdist(refDataPeripheral, inpDataPeripheral, 'euclidean')

            # plus epsilon
            for epsilon in range(1, limits + 1):
                index = epsilon + limits
                # d_{i,j} + d'_{i,j+epsilon}
                # d_{i,j}
                localCosts[:, :, index] += cdist(refDataBase, inpDataBase, 'euclidean')
                # d'_{i,j+epsilon}
                localCosts[:, :-epsilon, index] += cdist(refDataPeripheral, inpDataPeripheral[epsilon:], 'euclidean')
                # if j+epsilon is less than zero, then inf
                localCosts[:, -epsilon:, index] = np.inf
            """
            return localCosts

        def asyncCalc(refDatas, inpDatas):
            if len(refDatas) != 2 or len(inpDatas) != 2:
                raise ValueError('The length of both refDatas and inpDatas must be two, but got ref:{0} and inp:{1}'
                                 .format(len(refDatas), len(inpDatas)))
            R, I = refDatas[0].shape[0], inpDatas[0].shape[0]

            localCost = _async2LocalCost(refDatas[0], refDatas[1], inpDatas[0], inpDatas[1])
            matchingCost = np.zeros((R, I, 2*limits + 1))
            matchingCost[0, :, :] = localCost[0, :, :]
            #use slice instead of constraintInds = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
            for r in range(1, R):
                # i = 0
                for index, epsilon in enumerate(range(-limits, limits + 1)):
                    matchingCost[r, 0, index] = localCost[r, 0, index] + np.min(matchingCost[r - 1, 0, max(epsilon, 0):index + 1])
                # i = 1
                #for index, epsilon in enumerate(range(-limits, limits + 1)):
                    matchingCost[r, 1, index] = localCost[r, 1, index] + np.min(np.concatenate([matchingCost[r - 1, 1, max(epsilon, 0):index + 1],
                                                                                                matchingCost[r - 1, 0, max(epsilon + 1, 0):index + 2]]))

                # i = 2...
                #for index, epsilon in enumerate(range(-limits, limits + 1)):
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
            #inputFinFrameBackTracked, epsilonFinFrameBackTracked = kwargs['argsFinFrameBackTracked']
            inputFinFrameBackTracked, epsilonFinFrameBackTracked = np.unravel_index(np.nanargmin(matchingCost[matchingCost.shape[0] - 1]),
                                                                                                 matchingCost[matchingCost.shape[0] - 1].shape)

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

    elif kind == 'async3-lined-asynm':
        limits = 2
        def _async3LocalCost(refDataBase, refDataPeripheral1, refDataPeripheral2, inpDataBase, inpDataPeripheral1, inpDataPeripheral2):
            R, I = refDataBase.shape[0], inpDataBase.shape[0]

            localCosts = np.zeros((R, I, 2*limits + 1, 2*limits + 1)) # (r, i, epsilon_p1, epsilon_p2)

            refB_inpB_cdist = cdist(refDataBase, inpDataBase, 'euclidean')
            refP1_inpP1_cdist, refP2_inpP2_cdist = [], []
            for epsilon in range(-limits, 0):
                refP1_inpP1_cdist.append(cdist(refDataPeripheral1, inpDataPeripheral1[:epsilon], 'euclidean'))
                refP2_inpP2_cdist.append(cdist(refDataPeripheral2, inpDataPeripheral2[:epsilon], 'euclidean'))
            refP1_inpP1_cdist.append(cdist(refDataPeripheral1, inpDataPeripheral2, 'euclidean'))
            refP2_inpP2_cdist.append(cdist(refDataPeripheral2, inpDataPeripheral2, 'euclidean'))
            for epsilon in range(1, limits + 1):
                refP1_inpP1_cdist.append(cdist(refDataPeripheral1, inpDataPeripheral1[epsilon:], 'euclidean'))
                refP2_inpP2_cdist.append(cdist(refDataPeripheral2, inpDataPeripheral2[epsilon:], 'euclidean'))

            for index_p1, epsilon_p1 in enumerate(range(-limits, limits + 1)):
                for index_p2, epsilon_p2 in enumerate(range(-limits, limits + 1)):
                    localCosts[:, :, index_p1, index_p2] += refB_inpB_cdist

                    localCosts[:, -min(0, epsilon_p1):I + min(0, -epsilon_p1), index_p1, index_p2] += refP1_inpP1_cdist[index_p1]

                    localCosts[:, -min(0, epsilon_p2):I + min(0, -epsilon_p2), index_p1, index_p2] += refP2_inpP2_cdist[index_p2]

                    if epsilon_p1 < 0:
                        localCosts[:, :-epsilon_p1, index_p1, index_p2] = np.inf
                    elif epsilon_p1 > 0:
                        localCosts[:, -epsilon_p1:, index_p1, index_p2] = np.inf

                    if epsilon_p2 < 0:
                        localCosts[:, :-epsilon_p2, index_p1, index_p2] = np.inf
                    elif epsilon_p2 > 0:
                        localCosts[:, -epsilon_p2:, index_p1, index_p2] = np.inf

            return localCosts

        # both 1st refData and inpData will be treated as center point
        def asyncCalc(refDatas, inpDatas):
            if len(refDatas) != 3 or len(inpDatas) != 3:
                raise ValueError('The length of both refDatas and inpDatas must be three, but got ref:{0} and inp:{1}'
                                 .format(len(refDatas), len(inpDatas)))
            R, I = refDatas[0].shape[0], inpDatas[0].shape[0]

            localCost = _async3LocalCost(refDataBase=refDatas[0], refDataPeripheral1=refDatas[1], refDataPeripheral2=refDatas[2],
                                          inpDataBase=inpDatas[0], inpDataPeripheral1=inpDatas[1], inpDataPeripheral2=inpDatas[2])
            matchingCost = np.zeros((R, I, 2 * limits + 1, 2 * limits + 1))
            matchingCost[0, :, :, :] = localCost[0, :, :, :]

            for r in range(1, R):
                # i = 0
                for index_p1, epsilon_p1 in enumerate(range(-limits, limits + 1)):
                    for index_p2, epsilon_p2 in enumerate(range(-limits, limits + 1)):
                        matchingCost[r, 0, index_p1, index_p2] = localCost[r, 0, index_p1, index_p2] +\
                           np.min(matchingCost[r - 1, 0, max(epsilon_p1, 0):index_p1 + 1, max(epsilon_p2, 0):index_p2 + 1])

                        matchingCost[r, 1, index_p1, index_p2] = localCost[r, 1, index_p1, index_p2] + \
                           np.min(np.concatenate([matchingCost[r - 1, 1, max(epsilon_p1, 0):index_p1 + 1, max(epsilon_p2, 0):index_p2 + 1].flatten(),
                                                  matchingCost[r - 1, 0, max(epsilon_p1 + 1, 0):index_p1 + 2, max(epsilon_p2 + 1, 0):index_p2 + 2].flatten()]))

                        i_ = matchingCost[r - 1, 2:, max(epsilon_p1, 0):index_p1 + 1, max(epsilon_p2, 0):index_p2 + 1]
                        i_1 = matchingCost[r - 1, 1:-1, max(epsilon_p1 + 1, 0):index_p1 + 2, max(epsilon_p2 + 1, 0):index_p2 + 2]
                        i_2 = matchingCost[r - 1, :-2, max(epsilon_p1 + 2, 0):index_p1 + 3, max(epsilon_p2 + 2, 0):index_p2 + 3]
                        matchingCost[r, 2:, index_p1, index_p2] = localCost[r, 2:, index_p1, index_p2] + \
                           np.minimum.reduce(np.concatenate([i_.reshape((i_.shape[0], i_.shape[1] * i_.shape[2])),
                                                             i_1.reshape((i_1.shape[0], i_1.shape[1] * i_1.shape[2])),
                                                             i_2.reshape((i_2.shape[0], i_2.shape[1] * i_2.shape[2]))], axis=1), axis=1)

            if np.sum(np.isinf(matchingCost[R - 1, :, :, :])) == matchingCost.shape[1] * matchingCost.shape[2] * matchingCost.shape[3]: # all matching cost are infinity
                raise OverflowError('all matching cost are infinity')
            return matchingCost

        def asyncBackTrack(**kwargs):
            refDatas = kwargs['refDatas']
            inpDatas = kwargs['inpDatas']
            matchingCost = kwargs['matchingCost']

            #inputFinFrameBackTracked, epsilonFinFrameBackTracked = kwargs['argsFinFrameBackTracked']
            inputFinFrameBackTracked, epsilon_p1_FinFrameBackTracked, epsilon_p2_FinFrameBackTracked = \
                np.unravel_index(np.nanargmin(matchingCost[matchingCost.shape[0] - 1]),
                                              matchingCost[matchingCost.shape[0] - 1].shape)

            correspondentPointsBase, correspondentPointsPeripheral1, correspondentPointsPeripheral2 = [], [], []
            r, i, epsilon_p1, epsilon_p2 = matchingCost.shape[0] - 1, inputFinFrameBackTracked, \
                                            epsilon_p1_FinFrameBackTracked - limits, epsilon_p2_FinFrameBackTracked - limits

            correspondentPointsBase.append([r, i])
            correspondentPointsPeripheral1.append([r, i + epsilon_p1])
            correspondentPointsPeripheral2.append([r, i + epsilon_p2])

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
                    mcosts = [matchingCost[r - 1, i, max(epsilon_p1, 0):epsilon_p1 + limits + 1, max(epsilon_p2, 0):epsilon_p2 + limits + 1],
                              matchingCost[r - 1, i - 1, max(epsilon_p1 + 1, 0):epsilon_p1 + limits + 2, max(epsilon_p2 + 1, 0):epsilon_p2 + limits + 2],
                              matchingCost[r - 1, i - 2, max(epsilon_p1 + 2, 0):epsilon_p1 + limits + 3, max(epsilon_p2 + 2, 0):epsilon_p2 + limits + 3]]

                elif i == 1:
                    mcosts = [matchingCost[r - 1, 1, max(epsilon_p1, 0):epsilon_p1 + limits + 1, max(epsilon_p2, 0):epsilon_p2 + limits + 1],
                              matchingCost[r - 1, 0, max(epsilon_p1 + 1, 0):epsilon_p1 + limits + 2, max(epsilon_p2 + 1, 0):epsilon_p2 + limits + 2]]

                else: # i == 0
                    mcosts = [matchingCost[r - 1, 0, max(epsilon_p1, 0):epsilon_p1 + limits + 1, max(epsilon_p2, 0):epsilon_p2 + limits + 1]]

                c_p1, c_p2, tmp = [], [], []
                for ind, mcost in enumerate(mcosts):
                    cp1, cp2 = np.unravel_index(np.argmin(mcost), mcost.shape)
                    tmp.append(mcost[cp1, cp2])
                    c_p1.append(cp1)
                    c_p2.append(cp2)

                tmp = np.argmin(tmp)

                r = r - 1
                i = i - tmp
                epsilon_p1 = epsilon_p1 + c_p1[tmp] + forNewEpsilon[epsilon_p1 + limits][tmp]
                epsilon_p2 = epsilon_p2 + c_p2[tmp] + forNewEpsilon[epsilon_p2 + limits][tmp]
                correspondentPointsBase.insert(0, [r, i])
                correspondentPointsPeripheral1.insert(0, [r, i + epsilon_p1])
                correspondentPointsPeripheral2.insert(0, [r, i + epsilon_p2])


            return correspondentPointsBase, correspondentPointsPeripheral1, correspondentPointsPeripheral2

        return {'matchingCost': asyncCalc, 'backTrack': asyncBackTrack}

    elif kind == 'async2-visualization2' or kind == 'async2-localdiff': #or kind == 'async2-synm':
        limits = 2
        def _async2LocalCost(refDataBase, refDataPeripheral, inpDataBase, inpDataPeripheral):
            R, I = refDataBase.shape[0], inpDataBase.shape[0]

            localCosts = np.zeros((R, I, 2* 2 * limits + 1))  # (r, i, epsilon)
            for index, epsilon in enumerate(range(-limits*2, limits*2 + 1)):
                # d_{i,j} + d'_{i,j+epsilon}
                # d_{i,j}
                localCosts[:, :, index] += cdist(refDataBase, inpDataBase, 'euclidean')
                # d'_{i,j+epsilon}
                localCosts[:, -min(0, epsilon):I + min(0, -epsilon), index] += \
                    cdist(refDataPeripheral, inpDataPeripheral[max(0, epsilon):I - max(0, -epsilon)], 'euclidean')
                # if j+epsilon is no exist, then inf
                if epsilon < 0:
                    localCosts[:, :-epsilon, index] = np.inf
                elif epsilon > 0:
                    localCosts[:, -epsilon:, index] = np.inf

            return localCosts

        def async2Calc(refDatas, inpDatas):
            if len(refDatas) != 2 or len(inpDatas) != 2:
                raise ValueError('The length of both refDatas and inpDatas must be two, but got ref:{0} and inp:{1}'
                                 .format(len(refDatas), len(inpDatas)))
            R, I = refDatas[0].shape[0], inpDatas[0].shape[0]

            localCost = _async2LocalCost(refDatas[0], refDatas[1], inpDatas[0], inpDatas[1])
            matchingCost = np.zeros((R, I, 2 * 2 * limits + 1))
            matchingCost[0, :, :] = localCost[0, :, :]
            # initial conditions initial and fin frame correspond
            # this means all initial frame are inf excluding epsilon = 0
            matchingCost[0, :, :] = np.inf
            matchingCost[0, :, 2*limits] = localCost[0, :, 2*limits]

            matchingCost[1::2, :, :] = np.inf
            matchingCost[1:, 0, :] = np.inf
            # use slice instead of constraintInds = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
            for r in range(2, R, 2):
                for index, epsilon in enumerate(range(-2*limits, 2*limits + 1)):
                    # i = 1
                    matchingCost[r, 1, index] = localCost[r, 1, index] + np.min(
                        matchingCost[r - 2, 0, max(epsilon + 1, 0):index + 1])
                    # i = 2
                    matchingCost[r, 2, index] = localCost[r, 2, index] + np.min(
                        np.concatenate([matchingCost[r - 2, 1, max(epsilon + 1, 0):index + 1], # i-1
                                        matchingCost[r - 2, 0, max(epsilon + 2, 0):index + 2]])) # i-2
                    # i = 3
                    matchingCost[r, 3, index] = localCost[r, 3, index] + np.min(
                        np.concatenate([matchingCost[r - 2, 2, max(epsilon + 1, 0):index + 1],  # i-1
                                        matchingCost[r - 2, 1, max(epsilon + 2, 0):index + 2], #i-2
                                        matchingCost[r - 2, 0, max(epsilon + 3, 0):index + 3]]))  # i-3
                    # i > 3
                    matchingCost[r, 4:, index] = localCost[r, 4:, index] + np.minimum.reduce(np.concatenate(
                        [matchingCost[r - 2, 3:-1, max(epsilon + 1, 0):index + 1], #i-1
                         matchingCost[r - 2, 2:-2, max(epsilon + 2, 0):index + 2], #i-2
                         matchingCost[r - 2, 1:-3, max(epsilon + 3, 0):index + 3], # i-3
                         matchingCost[r - 2, :-4, max(epsilon + 4, 0):index + 4]], axis=1), axis=1) #i-4

            if R % 2 == 0:
                matchingCost[R - 1, :, :] = matchingCost[R - 2, :, :]

            if np.sum(np.isinf(matchingCost[R - 1, :, :])) == matchingCost.shape[1] * matchingCost.shape[2]:  # all matching cost are infinity
                raise OverflowError('all matching cost are infinity')
            return matchingCost

        def asyncBackTrack(**kwargs):
            refDatas = kwargs.get('refDatas')
            inpDatas = kwargs.get('inpDatas')
            matchingCost = kwargs.pop('matchingCost')
            #inputFinFrameBackTracked, epsilonFinFrameBackTracked = np.unravel_index(
            #    np.nanargmin(matchingCost[matchingCost.shape[0] - 1]),
            #    matchingCost[matchingCost.shape[0] - 1].shape)
            # finish condition
            inputFinFrameBackTracked = kwargs.get('inputFinFrameBackTracked', np.nanargmin(matchingCost[matchingCost.shape[0] - 1, :, 2*limits]))
            epsilonFinFrameBackTracked = 2*limits

            correspondentPointsBase, correspondentPointsPeripheral = [], []
            r, i, epsilon = matchingCost.shape[0] - 1, inputFinFrameBackTracked, epsilonFinFrameBackTracked - 2*limits

            if matchingCost.shape[0] % 2 == 0:
                r = r - 1
                correspondentPointsBase.insert(0, [r - 1, i])
                correspondentPointsPeripheral.insert(0, [r - 1, i + epsilon])
            else:
                correspondentPointsBase.insert(0, [r, i])
                correspondentPointsPeripheral.insert(0, [r, i + epsilon])

            """
            epsilon\i| i-1    i-2   i-3   i-4(tmp=0,1,2)
            -------------------------
                -4    | e+c   e+c   e+c   e+c
                -3    | e+c-1 e+c-1 e+c-1 e+c
                -2    | e+c-2 e+c-2 e+c-1 e+c
               -1~4   | e+c-3 e+c-2 e+c-1 e+c
            (epsilon + limits=0,...,4)
            """

            forNewEpsilon = [[0, 0, 0, 0],
                             [-1, -1, -1, 0],
                             [-2, -2, -1, 0],
                             [-3, -2, -1, 0],
                             [-3, -2, -1, 0],
                             [-3, -2, -1, 0],
                             [-3, -2, -1, 0],
                             [-3, -2, -1, 0],
                             [-3, -2, -1, 0],
                             [-3, -2, -1, 0]]

            while r > 0:
                index = epsilon + 2 * limits
                if i > 3:
                    mcosts = [matchingCost[r - 2, i - 1, max(epsilon + 1, 0):index + 1], #i-1
                                matchingCost[r - 2, i - 2, max(epsilon + 2, 0):index + 2], #i-2
                                matchingCost[r - 2, i - 3, max(epsilon + 3, 0):index + 3], # i-3
                                matchingCost[r - 2, i - 4, max(epsilon + 4, 0):index + 4]]

                elif i == 3:
                    mcosts = [matchingCost[r - 2, 2, max(epsilon + 1, 0):index + 1],  # i-1
                                matchingCost[r - 2, 1, max(epsilon + 2, 0):index + 2], #i-2
                                matchingCost[r - 2, 0, max(epsilon + 3, 0):index + 3]]

                elif i == 2:
                    mcosts = [matchingCost[r - 2, 1, max(epsilon + 1, 0):index + 1], # i-1
                                matchingCost[r - 2, 0, max(epsilon + 2, 0):index + 2]]
                elif i == 1:
                    mcosts = [matchingCost[r - 2, 0, max(epsilon + 1, 0):index + 1]]
                else:  # i == 0
                    break

                c_, tmp = [], []
                for ind, mcost in enumerate(mcosts):
                    c_.append(np.argmin(mcost))
                    tmp.append(mcost[c_[ind]])

                tmp = np.argmin(tmp)
                newepsilon = epsilon + c_[tmp] + forNewEpsilon[epsilon + 2*limits][tmp]
                tmp = tmp + 1
                if kind == 'async2-visualization2':
                    for tmp_i in range(1, tmp):
                        correspondentPointsBase.insert(0, [r - tmp_i * 2 / tmp, i - tmp_i])

                    # i + epsilon - (i - tmp + newepsilon) = epsilon + tmp - newepsilon means differences
                    diff = epsilon + tmp - newepsilon
                    for tmp_i in range(1, diff):
                        correspondentPointsPeripheral.insert(0, [r - tmp_i * 2 / diff, i + epsilon - tmp_i])

                r = r - 2
                i = i - tmp
                epsilon = newepsilon
                correspondentPointsBase.insert(0, [r, i])
                correspondentPointsPeripheral.insert(0, [r, i + epsilon])

            return correspondentPointsBase, correspondentPointsPeripheral

        return {'matchingCost': async2Calc, 'backTrack': asyncBackTrack}

    elif kind == 'async3-visualization2' or kind == 'async3-localdiff':
        # or kind == 'async2-synm':
        limits = 2
        def _async3LocalCost(refDataBase, refDataPeripheral1, refDataPeripheral2, inpDataBase, inpDataPeripheral1, inpDataPeripheral2):
            R, I = refDataBase.shape[0], inpDataBase.shape[0]

            localCosts = np.zeros((R, I, 2*2*limits + 1, 2*2*limits + 1)) # (r, i, epsilon_p1, epsilon_p2)

            refB_inpB_cdist = cdist(refDataBase, inpDataBase, 'euclidean')
            refP1_inpP1_cdist, refP2_inpP2_cdist = [], []
            for epsilon in range(-2*limits, 0):
                refP1_inpP1_cdist.append(cdist(refDataPeripheral1, inpDataPeripheral1[:epsilon], 'euclidean'))
                refP2_inpP2_cdist.append(cdist(refDataPeripheral2, inpDataPeripheral2[:epsilon], 'euclidean'))
            refP1_inpP1_cdist.append(cdist(refDataPeripheral1, inpDataPeripheral1, 'euclidean'))
            refP2_inpP2_cdist.append(cdist(refDataPeripheral2, inpDataPeripheral2, 'euclidean'))
            for epsilon in range(1, 2*limits + 1):
                refP1_inpP1_cdist.append(cdist(refDataPeripheral1, inpDataPeripheral1[epsilon:], 'euclidean'))
                refP2_inpP2_cdist.append(cdist(refDataPeripheral2, inpDataPeripheral2[epsilon:], 'euclidean'))

            for index_p1, epsilon_p1 in enumerate(range(-2*limits, 2*limits + 1)):
                for index_p2, epsilon_p2 in enumerate(range(-2*limits, 2*limits + 1)):
                    localCosts[:, :, index_p1, index_p2] += refB_inpB_cdist

                    localCosts[:, -min(0, epsilon_p1):I + min(0, -epsilon_p1), index_p1, index_p2] += refP1_inpP1_cdist[index_p1]

                    localCosts[:, -min(0, epsilon_p2):I + min(0, -epsilon_p2), index_p1, index_p2] += refP2_inpP2_cdist[index_p2]

                    if epsilon_p1 < 0:
                        localCosts[:, :-epsilon_p1, index_p1, index_p2] = np.inf
                    elif epsilon_p1 > 0:
                        localCosts[:, -epsilon_p1:, index_p1, index_p2] = np.inf

                    if epsilon_p2 < 0:
                        localCosts[:, :-epsilon_p2, index_p1, index_p2] = np.inf
                    elif epsilon_p2 > 0:
                        localCosts[:, -epsilon_p2:, index_p1, index_p2] = np.inf

            return localCosts

        # both 1st refData and inpData will be treated as center point
        def asyncCalc(refDatas, inpDatas):
            if len(refDatas) != 3 or len(inpDatas) != 3:
                raise ValueError('The length of both refDatas and inpDatas must be three, but got ref:{0} and inp:{1}'
                                 .format(len(refDatas), len(inpDatas)))
            R, I = refDatas[0].shape[0], inpDatas[0].shape[0]

            localCost = _async3LocalCost(refDataBase=refDatas[0], refDataPeripheral1=refDatas[1], refDataPeripheral2=refDatas[2],
                                          inpDataBase=inpDatas[0], inpDataPeripheral1=inpDatas[1], inpDataPeripheral2=inpDatas[2])
            matchingCost = np.zeros((R, I, 2*2 * limits + 1, 2*2 * limits + 1))
            matchingCost[0, :, :, :] = localCost[0, :, :, :]
            # initial conditions initial and fin frame correspond
            # this means all initial frame are inf excluding epsilon = 0
            matchingCost[0, :, :, :] = np.inf
            matchingCost[0, :, 2*limits, 2*limits] = localCost[0, :, 2*limits, 2*limits]

            matchingCost[1::2, :, :, :] = np.inf
            matchingCost[1:, 0, :, :] = np.inf
            for r in range(2, R, 2):
                # i = 0
                for index_p1, epsilon_p1 in enumerate(range(-2*limits, 2*limits + 1)):
                    for index_p2, epsilon_p2 in enumerate(range(-2*limits, 2*limits + 1)):
                        # i = 1
                        matchingCost[r, 1, index_p1, index_p2] = localCost[r, 1, index_p1, index_p2] +\
                            np.min(matchingCost[r - 2, 0, max(epsilon_p1 + 1, 0):index_p1 + 1, max(epsilon_p2 + 1, 0):index_p2 + 1])
                        # i = 2
                        matchingCost[r, 2, index_p1, index_p2] = localCost[r, 2, index_p1, index_p2] + np.min(
                            np.concatenate([matchingCost[r - 2, 1, max(epsilon_p1 + 1, 0):index_p1 + 1, max(epsilon_p2 + 1, 0):index_p2 + 1].flatten(),  # i-1
                                            matchingCost[r - 2, 0, max(epsilon_p1 + 2, 0):index_p1 + 2, max(epsilon_p2 + 2, 0):index_p2 + 2].flatten()]))  # i-2
                        # i = 3
                        matchingCost[r, 3, index_p1, index_p2] = localCost[r, 3, index_p1, index_p2] + np.min(
                            np.concatenate([matchingCost[r - 2, 2, max(epsilon_p1 + 1, 0):index_p1 + 1, max(epsilon_p2 + 1, 0):index_p2 + 1].flatten(),  # i-1
                                            matchingCost[r - 2, 1, max(epsilon_p1 + 2, 0):index_p1 + 2, max(epsilon_p2 + 2, 0):index_p2 + 2].flatten(),  # i-2
                                            matchingCost[r - 2, 0, max(epsilon_p1 + 3, 0):index_p1 + 3, max(epsilon_p2 + 3, 0):index_p2 + 3].flatten()]))  # i-3
                        # i > 3
                        i_1 = matchingCost[r - 2, 3:-1, max(epsilon_p1 + 1, 0):index_p1 + 1, max(epsilon_p2 + 1, 0):index_p2 + 1]  # i-1
                        i_2 = matchingCost[r - 2, 2:-2, max(epsilon_p1 + 2, 0):index_p1 + 2, max(epsilon_p2 + 2, 0):index_p2 + 2]  # i-2
                        i_3 = matchingCost[r - 2, 1:-3, max(epsilon_p1 + 3, 0):index_p1 + 3, max(epsilon_p2 + 3, 0):index_p2 + 3]  # i-3
                        i_4 = matchingCost[r - 2, :-4, max(epsilon_p1 + 4, 0):index_p1 + 4, max(epsilon_p2 + 4, 0):index_p2 + 4]

                        matchingCost[r, 4:, index_p1, index_p2] = localCost[r, 4:, index_p1, index_p2] + \
                           np.minimum.reduce(np.concatenate([i_1.reshape((i_1.shape[0], i_1.shape[1] * i_1.shape[2])),
                                                             i_2.reshape((i_2.shape[0], i_2.shape[1] * i_2.shape[2])),
                                                             i_3.reshape((i_3.shape[0], i_3.shape[1] * i_3.shape[2])),
                                                             i_4.reshape((i_4.shape[0], i_4.shape[1] * i_4.shape[2]))], axis=1), axis=1)

            if R % 2 == 0:
                matchingCost[R - 1, :, :, :] = matchingCost[R - 2, :, :, :]

            if np.sum(np.isinf(matchingCost[R - 1, :, :, :])) == matchingCost.shape[1] * matchingCost.shape[2] * matchingCost.shape[3]: # all matching cost are infinity
                raise OverflowError('all matching cost are infinity')
            return matchingCost

        def asyncBackTrack(**kwargs):
            refDatas = kwargs.get('refDatas')
            inpDatas = kwargs.get('inpDatas')
            matchingCost = kwargs.pop('matchingCost')

            #inputFinFrameBackTracked, epsilon_p1_FinFrameBackTracked, epsilon_p2_FinFrameBackTracked = \
            #    np.unravel_index(np.nanargmin(matchingCost[matchingCost.shape[0] - 1]),
            #                                  matchingCost[matchingCost.shape[0] - 1].shape)
            inputFinFrameBackTracked = kwargs.get('inputFinFrameBackTracked', np.nanargmin(matchingCost[matchingCost.shape[0] - 1, :, 2*limits, 2*limits]))
            epsilon_p1_FinFrameBackTracked, epsilon_p2_FinFrameBackTracked = 2*limits, 2*limits

            correspondentPointsBase, correspondentPointsPeripheral1, correspondentPointsPeripheral2 = [], [], []
            r, i, epsilon_p1, epsilon_p2 = matchingCost.shape[0] - 1, inputFinFrameBackTracked, \
                                            epsilon_p1_FinFrameBackTracked - 2*limits, epsilon_p2_FinFrameBackTracked - 2*limits

            if matchingCost.shape[0] % 2 == 0:
                r = r - 1
                correspondentPointsBase.insert(0, [r - 1, i])
                correspondentPointsPeripheral1.insert(0, [r - 1, i + epsilon_p1])
                correspondentPointsPeripheral2.insert(0, [r - 1, i + epsilon_p2])
            else:
                correspondentPointsBase.insert(0, [r, i])
                correspondentPointsPeripheral1.insert(0, [r, i + epsilon_p1])
                correspondentPointsPeripheral2.insert(0, [r, i + epsilon_p2])

            """
            epsilon\i| i-1    i-2   i-3   i-4(tmp=0,1,2)
            -------------------------
                -4    | e+c   e+c   e+c   e+c
                -3    | e+c-1 e+c-1 e+c-1 e+c
                -2    | e+c-2 e+c-2 e+c-1 e+c
               -1~4   | e+c-3 e+c-2 e+c-1 e+c
            (epsilon + limits=0,...,4)
            """

            forNewEpsilon = [[0, 0, 0, 0],
                             [-1, -1, -1, 0],
                             [-2, -2, -1, 0],
                             [-3, -2, -1, 0],
                             [-3, -2, -1, 0],
                             [-3, -2, -1, 0],
                             [-3, -2, -1, 0],
                             [-3, -2, -1, 0],
                             [-3, -2, -1, 0],
                             [-3, -2, -1, 0]]

            while r > 0:
                index_p1, index_p2 = epsilon_p1 + 2 * limits, epsilon_p2 + 2 * limits
                if i > 3:
                    mcosts = [matchingCost[r - 2, i - 1, max(epsilon_p1 + 1, 0):index_p1 + 1, max(epsilon_p2 + 1, 0):index_p2 + 1],  # i-1
                              matchingCost[r - 2, i - 2, max(epsilon_p1 + 2, 0):index_p1 + 2, max(epsilon_p2 + 2, 0):index_p2 + 2],  # i-2
                              matchingCost[r - 2, i - 3, max(epsilon_p1 + 3, 0):index_p1 + 3, max(epsilon_p2 + 3, 0):index_p2 + 3],  # i-3
                              matchingCost[r - 2, i - 4, max(epsilon_p1 + 4, 0):index_p1 + 4, max(epsilon_p2 + 4, 0):index_p2 + 4]]

                elif i == 3:
                    mcosts = [matchingCost[r - 2, 2, max(epsilon_p1 + 1, 0):index_p1 + 1, max(epsilon_p2 + 1, 0):index_p2 + 1],  # i-1
                              matchingCost[r - 2, 1, max(epsilon_p1 + 2, 0):index_p1 + 2, max(epsilon_p2 + 2, 0):index_p2 + 2],  # i-2
                              matchingCost[r - 2, 0, max(epsilon_p1 + 3, 0):index_p1 + 3, max(epsilon_p2 + 3, 0):index_p2 + 3]]

                elif i == 2:
                    mcosts = [matchingCost[r - 2, 1, max(epsilon_p1 + 1, 0):index_p1 + 1, max(epsilon_p2 + 1, 0):index_p2 + 1],  # i-1
                              matchingCost[r - 2, 0, max(epsilon_p1 + 2, 0):index_p1 + 2, max(epsilon_p2 + 2, 0):index_p2 + 2]]
                elif i == 1:
                    mcosts = [matchingCost[r - 2, 0, max(epsilon_p1 + 1, 0):index_p1 + 1, max(epsilon_p2 + 1, 0):index_p2 + 1]]
                else:  # i == 0
                    break

                c_p1, c_p2, tmp = [], [], []
                for ind, mcost in enumerate(mcosts):
                    cp1, cp2 = np.unravel_index(np.argmin(mcost), mcost.shape)
                    tmp.append(mcost[cp1, cp2])
                    c_p1.append(cp1)
                    c_p2.append(cp2)

                tmp = np.argmin(tmp)
                newepsilon_p1 = epsilon_p1 + c_p1[tmp] + forNewEpsilon[epsilon_p1 + 2 * limits][tmp]
                newepsilon_p2 = epsilon_p2 + c_p2[tmp] + forNewEpsilon[epsilon_p2 + 2 * limits][tmp]
                tmp = tmp + 1
                if kind == 'async3-visualization2':
                    for tmp_i in range(1, tmp):
                        correspondentPointsBase.insert(0, [r - tmp_i * 2 / tmp, i - tmp_i])

                    # i + epsilon - (i - tmp + newepsilon) = epsilon + tmp - newepsilon means differences
                    diff = epsilon_p1 + tmp - newepsilon_p1
                    for tmp_i in range(1, diff):
                        correspondentPointsPeripheral1.insert(0, [r - tmp_i * 2 / diff, i + epsilon_p1 - tmp_i])

                    diff = epsilon_p2 + tmp - newepsilon_p2
                    for tmp_i in range(1, diff):
                        correspondentPointsPeripheral2.insert(0, [r - tmp_i * 2 / diff, i + epsilon_p2 - tmp_i])

                r = r - 2
                i = i - tmp
                epsilon_p1 = newepsilon_p1
                epsilon_p2 = newepsilon_p2
                correspondentPointsBase.insert(0, [r, i])
                correspondentPointsPeripheral1.insert(0, [r, i + epsilon_p1])
                correspondentPointsPeripheral2.insert(0, [r, i + epsilon_p2])

            return correspondentPointsBase, correspondentPointsPeripheral1, correspondentPointsPeripheral2

        return {'matchingCost': asyncCalc, 'backTrack': asyncBackTrack}

    else:
        raise NameError('{0} is invalid constraint name'.format(kind))