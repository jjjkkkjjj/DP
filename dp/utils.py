import os
import csv
import sys
import numpy as np
from .dp import DP
from .data import Data

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

def csvReader(csvfile, dir):

    tmplists = []

    with open(os.path.join(dir, csvfile), "r") as f:
        reader = csv.reader(f)
        for row in reader:
            tmplists.append(row)

    time = len(tmplists) - 7
    data = Data(interpolate='linear')

    jointNames = []
    X = []
    Y = []
    Z = []

    index = 2
    while index < len(tmplists[6]):
        if tmplists[5][index] == 'Position':
            if tmplists[2][index] == 'Bone Marker' and tmplists[3][index].find('Unlabeled') < 0:
                x = np.array(
                    [tmplists[i][index] if tmplists[i][index] != '' else np.nan for i in
                     range(7, len(tmplists))],
                    dtype='float')
                y = np.array([tmplists[i][index + 1] if tmplists[i][index + 1] != '' else np.nan for i in
                              range(7, len(tmplists))], dtype='float')
                z = np.array([tmplists[i][index + 2] if tmplists[i][index + 2] != '' else np.nan for i in
                              range(7, len(tmplists))], dtype='float')
                X.append(x)
                Y.append(y)
                Z.append(z)
                jointNames.append(tmplists[3][index])
            index += 3
        elif tmplists[5][index] == 'Rotation':
            index += 4

    data.setvalues(csvfile, np.array(X), np.array(Y), np.array(Z), jointNames, dir=dir, lines='baseball')

    return data