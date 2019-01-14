import numpy as np

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


    else:
        raise ValueError('{0} is invalid constraint name'.format(kind))