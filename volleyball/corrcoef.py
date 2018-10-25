import csv
from dp.dp import DP, referenceReader, Data
import numpy as np
from dp.view import Visualization
from dp.calccorrcoef import corrcoefMean
import os
import sys

def main():
    name = "IMAMURA"
    Dir = "./trc/" + name

    resultSuperDir = os.path.join('./result/', name)
    if not os.path.exists(resultSuperDir):
        os.mkdir(resultSuperDir)
    resultSuperDir = os.path.join(resultSuperDir, 'corrcoef')
    if not os.path.exists(resultSuperDir):
        os.mkdir(resultSuperDir)

    reffile = referenceReader("{0}-normal.csv".format(name), Dir, superDir=name)
    refData = Data()
    refData.set_from_trc(os.path.join(Dir, reffile), lines='volleyball')

    corrcoef = writecorrcoef(name, 1, 36 + 1, showcorrcoef=True)
    for i in range(1, 7 + 1):
        sys.stdout.write("\rcalculating now... {0}/{1}".format(i, 8))
        sys.stdout.flush()
        filename = '{0}{1:02d}.trc'.format(name, i)

        if filename in reffile:
            continue

        inpData = Data()
        inpData.set_from_trc(os.path.join('./trc', name, filename), lines='volleyball')

        DP_ = DP(refData, inpData, verbose=False, ignoreWarning=True)

        resultDir = os.path.join(resultSuperDir, filename[:-4])
        if not os.path.exists(resultDir):
            os.mkdir(resultDir)
        DP_.calc_corrcoef(corrcoef, showresult=False, resultdir=resultDir, correspondLine=True)
    exit()

def writecorrcoef(name, rangeMin, rangeMax, showcorrcoef):
    Datalists = []
    Dir = "./trc/" + name
    for i in range(rangeMin, rangeMax):
        filename = name + '{0:02d}.trc'.format(i)
        data = Data()
        data.set_from_trc(os.path.join(Dir, filename), lines='volleyball')
        Datalists.append(data)
    return corrcoefMean(Datalists, verbose=False, showcorrcoef=showcorrcoef)


def comp_independent_corrcoef():
    view = Visualization()

    name = "IMAMURA"
    Dir = "./trc/" + name

    resultSuperDir = os.path.join('./result/', name)
    if not os.path.exists(resultSuperDir):
        os.mkdir(resultSuperDir)
    resultSuperDir = os.path.join(resultSuperDir, 'comp_ind-corr')
    if not os.path.exists(resultSuperDir):
        os.mkdir(resultSuperDir)

    reffile = referenceReader("{0}-normal.csv".format(name), Dir, superDir=name)
    refData = Data()
    refData.set_from_trc(os.path.join(Dir, reffile), lines='volleyball')

    corrcoef = writecorrcoef(name, 1, 7 + 1, showcorrcoef=True)
    for i in range(1, 7 + 1):
        sys.stdout.write("\rcalculating now... {0}/{1}".format(i, 8))
        sys.stdout.flush()
        filename = '{0}{1:02d}.trc'.format(name, i)

        if filename in reffile:
            continue

        inpData = Data()
        inpData.set_from_trc(os.path.join('./trc', name, filename), lines='volleyball')

        DP_ = DP(refData, inpData, verbose=False, ignoreWarning=True, verboseNan=False)

        resultDir = os.path.join(resultSuperDir, filename[:-4])
        if not os.path.exists(resultDir):
            os.mkdir(resultDir)

        DP_.calc(showresult=False)
        indeopendentX, indeopendentY = DP_.resultData()

        DP_.calc_corrcoef(corrcoef, showresult=False)
        corrcoefX, corrcoefY = DP_.resultData()

        for joint in refData.joints.keys():
            if joint not in DP_.correspondents.keys():
                continue
            X = {}
            Y = {}

            X['independent'] = indeopendentX[joint]
            Y['independent'] = indeopendentY[joint]
            X['correlation efficient'] = corrcoefX[joint]
            Y['correlation efficient'] = corrcoefY[joint]
            view.show(x=X, y=Y, xtime=refData.frame_max, ytime=inpData.frame_max,
                  title=joint, legend=True, savepath=resultDir + "/{0}-R_{1}-I_{2}.png".format(joint, refData.name, inpData.name))
    pass

if __name__ == '__main__':
    #main()
    comp_independent_corrcoef()
    #writecorrcoef()