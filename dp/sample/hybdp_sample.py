import numpy as np
import itertools
import sys
import scipy.sparse as sp
from scipy.spatial.distance import cdist

refData = {'a':np.array([1,2,3,4,5,6]),
           'b':np.array([1,2,3,3,4,5]),
           'c':np.array([1,2,3,4,5,6]),
           'd':np.array([1,2,3,4,5,6]),
           'e':np.array([1,2,3,4,5,6]),
           'f':np.array([1,2,4,5,6,6]),
           'g':np.array([1,2,4,5,6,6])}

inpData = {'a':np.array([1,1,1,2,3,4,5,6]),
           'b':np.array([1,2,3,3,4,5,5,5]),
           'c':np.array([1,2,3,4,5,6,6,6]),
           'd':np.array([1,1,2,3,4,5,6,6]),
           'e':np.array([1,2,2,3,4,5,6,6]),
           'f':np.array([1,2,2,3,4,5,6,6]),
           'g':np.array([1,2,2,3,4,5,6,6])}

refFrameMax = refData['a'].shape[0]
inpFrameMax = inpData['a'].shape[0]

refData = {joint:data[:, np.newaxis] for joint, data in refData.items()}
inpData = {joint:data[:, np.newaxis] for joint, data in inpData.items()}

joints = list(refData.keys())

def hybdp(Bases, Neighbors, Epsilon):

    epsilon = None
    if isinstance(Epsilon, int):
        epsilon = np.arange(Epsilon)
    elif isinstance(Epsilon, list) or isinstance(Epsilon, tuple):
        epsilon = np.arange(*Epsilon)
    else:
        raise ValueError('epsilon must be int or array, but {0}'.format(type(Epsilon).__name__))

    if len(Bases) != len(Neighbors):
        raise ValueError('Length of both Bases {0} and Neighbors {1} must be same'.format(Bases, Neighbors))
    _check(Bases, Neighbors)

    coefs = _preprocess(Bases, Neighbors)

    # set linear programming with totality unimodular constraint matrix
    # linear interger programming
    D = {}
    for base, neighbors in zip(Bases, Neighbors):
        #D_ = {}
        # D_'s shape is (reft, inpt, epsilon)
        D_ = np.zeros((refFrameMax * len(neighbors), inpFrameMax, len(epsilon)))
        for index, neighbor in enumerate(neighbors):
            costMatBase = cdist(refData[base], inpData[base], 'euclidean')
            costMatNeighbor = cdist(refData[neighbor], inpData[neighbor], 'euclidean')
            costMat = []
            # row is ref index, column is inp index
            for eps in epsilon:
                if eps < 0:
                    costMat.append(
                        costMatBase / coefs[base] + np.hstack([np.ones((costMatNeighbor.shape[0], -eps))*np.inf,
                                                                costMatNeighbor[:, :eps]]) / coefs[neighbor])
                elif eps == 0:
                    costMat.append(
                        costMatBase / coefs[base] + costMatNeighbor / coefs[neighbor])
                else:
                    costMat.append(
                        costMatBase / coefs[base] + np.hstack([costMatNeighbor[:, eps:],
                                                               np.ones((costMatNeighbor.shape[0], eps)) * np.inf]) / coefs[neighbor])
            #D_[neighbor] = np.array(costMat)
            # D_'s shape is (reft, inpt, epsilon)
            D_[index::len(neighbors), :] = np.array(costMat).transpose((1, 2, 0))

        D[base] = D_

    """
    start hybdp
    initialization:
    
    """
    baseCulm = np.zeros((refFrameMax, inpFrameMax))


def _check(Bases, Neighbors):
    """
    uppper triangle matrix of adjancency matrix and
    lower triangle matrix of it must be same
    In case we focus on neighbor which includes base only
    """
    adjBase = np.zeros((len(joints), len(joints)))
    try:
        BasesNum = [joints.index(base) for base in Bases]
        NeighborsNum = [[joints.index(neighbor) for neighbor in neighbors] for neighbors in Neighbors]
    except ValueError as e:
        raise ValueError('{0} included in joints'.format(e.args[0][:-8]))

    for basenum, neighborsnum in zip(BasesNum, NeighborsNum):
        adjBase[basenum, list(set(neighborsnum) & set(BasesNum))] = 1

    check = np.triu(adjBase) != np.tril(adjBase).T
    if np.sum(check) > 0:
        baseind, neighborind = np.where(check)
        raise ValueError('neighbor {0} for base {1} must be included'
                         .format(np.array(joints)[baseind], np.array(joints)[neighborind]))


def _preprocess(Bases, Neighbors):
    coefs = {joint: 0 for joint in joints}

    for base, neighbors in zip(Bases, Neighbors):
        coefs[base] = len(neighbors)
        for neighbor in list(set(neighbors) - set(Bases)):
            coefs[neighbor] = 1


    return coefs



if __name__ == '__main__':
    hybdp(['a', 'd', 'e'],
          [['b', 'c', 'd'], ['e', 'a', 'g'], ['d', 'f']],
          (-1,2))