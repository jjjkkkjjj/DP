import csv
from dp.dp import DP, referenceReader, Data
from csvReader import csvReader
import numpy as np
from dp.view import Visualization
from dp.calccorrcoef import corrcoefMean
import os
import sys

def main():

    with open('pitch-type.csv', 'r') as f:
        reader = csv.reader(f)
        for type in reader:
            # read data
            csvfiles = type[2:]
            Dir = type[0]
            pitchType = type[1]
            name = Dir.split('/')[-2]

            resultSuperDir = os.path.join('./result/', name)
            if not os.path.exists(resultSuperDir):
                os.mkdir(resultSuperDir)
            resultSuperDir = os.path.join(resultSuperDir, '{0}-corr'.format(pitchType))
            if not os.path.exists(resultSuperDir):
                os.mkdir(resultSuperDir)

            reffile = referenceReader("{0}-{1}.csv".format(name, pitchType), Dir, superDir=name)
            refData = csvReader(csvfiles[csvfiles.index(reffile)], Dir)

            corrcoef = writecorrcoef(Dir, csvfiles, showcorrcoef=True)
            for i, csvfile in enumerate(csvfiles):
                sys.stdout.write("\rcalculating now... {0}/{1}".format(i, len(csvfiles)))
                sys.stdout.flush()
                if csvfile == reffile:
                    continue

                inpData = csvReader(csvfile, Dir)

                DP_ = DP(refData, inpData, verbose=False, ignoreWarning=True)

                resultDir = os.path.join(resultSuperDir, csvfile[:-4])
                if not os.path.exists(resultDir):
                    os.mkdir(resultDir)
                DP_.calc_corrcoef(corrcoef, showresult=False, resultdir=resultDir, correspondLine=True)
            exit()


def writecorrcoef(Dir, csvfiles, showcorrcoef):
    Datalists = []
    for csvfile in csvfiles:
        data = csvReader(csvfile, Dir)
        Datalists.append(data)
    return corrcoefMean(Datalists, verbose=False, showcorrcoef=showcorrcoef)


def comp_independent_corrcoef():
    view = Visualization()

    with open('pitch-type.csv', 'r') as f:
        reader = csv.reader(f)
        for type in reader:
            # read data
            csvfiles = type[2:]
            Dir = type[0]
            pitchType = type[1]
            name = Dir.split('/')[-2]

            resultSuperDir = os.path.join('./result/', name)
            if not os.path.exists(resultSuperDir):
                os.mkdir(resultSuperDir)
            resultSuperDir = os.path.join(resultSuperDir, '{0}-comp_ind-corr'.format(pitchType))
            if not os.path.exists(resultSuperDir):
                os.mkdir(resultSuperDir)

            reffile = referenceReader("{0}-{1}.csv".format(name, pitchType), Dir, superDir=name)
            refData = csvReader(csvfiles[csvfiles.index(reffile)], Dir)

            corrcoef = writecorrcoef(Dir, csvfiles, showcorrcoef=True)
            for i, csvfile in enumerate(csvfiles):
                sys.stdout.write("\rcalculating now... {0}/{1}".format(i, len(csvfiles)))
                sys.stdout.flush()
                if csvfile == reffile:
                    continue

                inpData = csvReader(csvfile, Dir)

                DP_ = DP(refData, inpData, verbose=False, ignoreWarning=True)

                resultDir = os.path.join(resultSuperDir, csvfile[:-4])
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
                              title=joint, legend=True,
                              savepath=resultDir + "/{0}-R_{1}-I_{2}.png".format(joint, refData.name, inpData.name))
            exit()


if __name__ == '__main__':
    #main()
    comp_independent_corrcoef()
    #writecorrcoef()