import csv
from dp.contextdp import SyncContextDP, AsyncContextDP
from dp.utils import referenceReader, csvReader
import numpy as np
from dp.view import Visualization
from dp.calccorrcoef import corrcoefMean
import os
import sys

def implementDP(type, method):
    # read data
    csvfiles = type[2:]
    Dir = type[0]
    pitchType = type[1]
    name = Dir.split('/')[-2]

    resultSuperDir = os.path.join('./result/', name)
    if not os.path.exists(resultSuperDir):
        os.mkdir(resultSuperDir)
    #resultSuperDir = os.path.join(resultSuperDir, '{0}-corr'.format(pitchType))
    resultSuperDir = os.path.join(resultSuperDir, '{0}-{1}'.format(pitchType, method))
    if not os.path.exists(resultSuperDir):
        os.mkdir(resultSuperDir)

    reffile = referenceReader("{0}-{1}.csv".format(name, pitchType), Dir, superDir=name)
    refData = csvReader(csvfiles[csvfiles.index(reffile)], Dir)

    """
    lines = [['LTOE', 'LANK'], ['LTIB', 'LANK'], ['LASI', 'LPSI'],  # around ankle
            ['RTOE', 'RANK'], ['RTIB', 'RANK'], ['RASI', 'RPSI'],  # "
            ['LASI', 'RASI'], ['LPSI', 'RPSI'], ['LHEE', 'LANK'], ['RHEE', 'RANK'], ['LHEE', 'LTOE'], ['RHEE', 'RTOE'],  # around hip
            ['LHEE', 'LTIB'], ['RHEE', 'RTIB'],  # connect ankle to knee
            ['LKNE', 'LTIB'], ['LKNE', 'LTHI'], ['LASI', 'LTHI'], ['LPSI', 'LTHI'],  # connect knee to hip
            ['RKNE', 'RTIB'], ['RKNE', 'RTHI'], ['RASI', 'RTHI'], ['RPSI', 'RTHI'],  # "
            ['LPSI', 'T10'], ['RPSI', 'T10'], ['LASI', 'STRN'], ['RASI', 'STRN'],  # conncet lower and upper
            # upper
            ['LFHD', 'LBHD'], ['RFHD', 'RBHD'], ['LFHD', 'RFHD'], ['LBHD', 'RBHD'],  # around head
            ['LBHD', 'C7'], ['RBHD', 'C7'], ['C7', 'CLAV'], ['CLAV', 'LSHO'], ['CLAV', 'RSHO'],# connect head to shoulder
            ['LSHO', 'LBAK'], ['RSHO', 'RBAK'], ['RBAK', 'LBAK'],  # around shoulder
            ['LWRA', 'LFIN'], ['LWRA', 'LFIN'], ['LWRA', 'LWRB'], ['LWRA', 'LFRM'], ['LWRB', 'LFRM'],# around wrist
            ['RWRA', 'RFIN'], ['RWRA', 'RFIN'], ['RWRA', 'RWRB'], ['RWRA', 'RFRM'], ['RWRB', 'RFRM'],  # "
            ['LELB', 'LRFM'], ['LELB', 'LUPA'], ['LELB', 'LFIN'], ['LUPA', 'LSHO'],# connect elbow to wrist, connect elbow to shoulder
            ['RELB', 'RRFM'], ['RELB', 'RUPA'], ['RELB', 'RFIN'], ['RUPA', 'RSHO'],  # "
            ['LSHO', 'STRN'], ['RSHO', 'STRN'], ['LBAK', 'T10'], ['RBAK', 'T10'],# connect shoulder to torso
            ]
    """
    contexts = [['RFHD', 'LFHD'], ['RBHD', 'LBHD'],
                ['T10', 'RBAK'],
                ['CLAV', 'RSHO', 'LSHO'],
                ['RELB', 'RUPA', 'RFRM'], ['LELB', 'LUPA', 'LFRM'],
                ['RFIN', 'RWRA', 'RWRB'], ['LFIN', 'LWRA', 'LWRB'],
                ['LPSI', 'RPSI'],
                ['RTHI', 'RASI', 'RKNE'], ['LTHI', 'LASI', 'LKNE'],
                ['RANK', 'RTOE', 'RTIB'], ['LANK', 'LTOE', 'LTIB']]
    tmp = []
    for context in contexts:
        tmp_ = []
        for joint in context:
            tmp_.append('Skeleton 02:{0}'.format(joint))
        tmp.append(tmp_)
    contexts = tmp

    kinds = ['async2-visualization2', 'async2-visualization2',
             'async2-visualization2',
             'async3-visualization2',
             'async3-visualization2', 'async3-visualization2',
             'async3-visualization2', 'async3-visualization2',
             'async2-visualization2',
             'async3-visualization2', 'async3-visualization2',
             'async3-visualization2', 'async3-visualization2']

    for i, csvfile in enumerate(csvfiles):
        sys.stdout.write("\rcalculating now... {0}/{1}".format(i, len(csvfiles)))
        sys.stdout.flush()
        if csvfile == reffile:
            continue

        inpData = csvReader(csvfile, Dir)

        if 'async' in method: # async context dp
            DP_ = AsyncContextDP(contexts=contexts, reference=refData, input=inpData, verbose=False, ignoreWarning=True,
                                verboseNan=False)
        else:
            DP_ = SyncContextDP(contexts=contexts, reference=refData, input=inpData, verbose=False, ignoreWarning=True, verboseNan=False)

        resultDir = os.path.join(resultSuperDir, csvfile[:-4])
        if not os.path.exists(resultDir):
            os.mkdir(resultDir)

        if method == 'independent':
            DP_.synchronous()
            view = Visualization()
            X, Y = DP_.resultData()
            view.show(x=X, y=Y, xtime=refData.frame_max, ytime=inpData.frame_max,
                      title='overlayed all matching costs', legend=True, correspondLine=False,
                      savepath=resultDir + "/overlayed-all-matching-costs.png")

        elif method == 'sync-visualization':
            fps = 240
            colors = DP_.resultVisualization(kind='visualization', fps=fps, maximumGapTime=0.1, resultDir=resultDir)
            DP_.input.show(fps=fps, colors=colors)
            DP_.input.save(path=resultDir + "/R_{0}-I_{1}.mp4".format(refData.name, inpData.name), fps=fps, colors=colors, saveonly=True)

        elif method == 'sync-visualization2':
            fps = 240
            colors = DP_.resultVisualization(kind='visualization2', fps=fps, maximumGapTime=0.1, resultDir=resultDir)
            DP_.input.show(fps=fps, colors=colors)
            DP_.input.save(path=resultDir + "/R_{0}-I_{1}.mp4".format(refData.name, inpData.name), fps=fps, colors=colors, saveonly=True)


        elif method == 'async':
            DP_.asynchronous(kinds=kinds)
            view = Visualization()
            X, Y = DP_.resultData()
            view.show(x=X, y=Y, xtime=refData.frame_max, ytime=inpData.frame_max,
                      title='overlayed all matching costs', legend=True, correspondLine=False,
                      savepath=resultDir + "/overlayed-all-matching-costs.png")
            exit()

        elif method == 'async-visualization2':
            fps = 240
            colors = DP_.resultVisualization(kinds=kinds, kind='visualization2', fps=fps, maximumGapTime=0.1, resultDir=resultDir)
            DP_.input.show(fps=fps, colors=colors)
            exit()
            DP_.input.save(path=resultDir + "/R_{0}-I_{1}.mp4".format(refData.name, inpData.name), fps=fps, colors=colors, saveonly=True)

        else:
            raise ValueError("{0} is invalid method".format(method))

    print("\nfinished {0}-{1}".format(name, pitchType))

def main(method):

    with open('pitch-type.csv', 'r') as f:
        reader = csv.reader(f)
        for type in reader:
            implementDP(type=type, method=method)

if __name__ == '__main__':
    # main(method='sync')
    # main(method='sync-visualization')
    # main(method='sync-visualization2')
    # main(method='async')
    main(method='async-visualization2')