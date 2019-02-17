from dp.contextdp import SyncContextDP, AsyncContextDP
from dp.dp import DP
from dp.view import Visualization
from dp.data import Data
from dp.utils import csvReader
import os
import sys


def implementDP(dir, refname, inpname, refini, reffin, inpini, inpfin):
    # read data
    name = os.path.basename(os.path.dirname(dir))

    resultSuperDir = os.path.join('./compareDP/', name)
    if not os.path.exists(resultSuperDir):
        os.mkdir(resultSuperDir)
    # resultSuperDir = os.path.join(resultSuperDir, '{0}-corr'.format(pitchType))
    resultSuperDir = os.path.join(resultSuperDir, 'r{0}-i{1}'.format(refname.split('.')[0], inpname.split('.')[0]))
    if not os.path.exists(resultSuperDir):
        os.mkdir(resultSuperDir)

    refData = csvReader(refname, dir)
    # rename joint
    for joint in list(refData.joints.keys()):
        refData.joints[joint.split(':')[-1]] = refData.joints.pop(joint)
    refData.cutFrame(ini=refini, fin=reffin, save=True, update=False)

    inpData = csvReader(inpname, dir)
    # rename joint
    for joint in list(inpData.joints.keys()):
        inpData.joints[joint.split(':')[-1]] = inpData.joints.pop(joint)
    inpData.cutFrame(ini=inpini, fin=inpfin, save=True, update=False)

    contexts = [['RFHD', 'LFHD'], ['RBHD', 'LBHD'],
                ['T10', 'RBAK'],
                ['CLAV', 'RSHO', 'LSHO'],
                ['RELB', 'RUPA', 'RFRM'], ['LELB', 'LUPA', 'LFRM'],
                ['RFIN', 'RWRA', 'RWRB'], ['LFIN', 'LWRA', 'LWRB'],
                ['LPSI', 'RPSI'],
                ['RTHI', 'RASI', 'RKNE'], ['LTHI', 'LASI', 'LKNE'],
                ['RANK', 'RTOE', 'RTIB'], ['LANK', 'LTOE', 'LTIB']]

    kinds = ['async2-visualization2', 'async2-visualization2',
             'async2-visualization2',
             'async3-visualization2',
             'async3-visualization2', 'async3-visualization2',
             'async3-visualization2', 'async3-visualization2',
             'async2-visualization2',
             'async3-visualization2', 'async3-visualization2',
             'async3-visualization2', 'async3-visualization2']


    # independent
    DP_ = DP(refData, inpData, verbose=False, ignoreWarning=True, verboseNan=False)
    DP_.resultVisualization(kind='visualization2', fps=500, maximumGapTime=0.1)
    Xi, Yi = DP_.resultData()

    # sync
    DP_ = SyncContextDP(contexts=contexts, reference=refData, input=inpData, verbose=False, ignoreWarning=True,
                        verboseNan=False)
    DP_.resultVisualization(kind='visualization2', fps=500, maximumGapTime=0.1)
    Xs, Ys = DP_.resultData()

    # async
    DP_ = AsyncContextDP(contexts=contexts, reference=refData, input=inpData, verbose=False, ignoreWarning=True,
                         verboseNan=False)
    DP_.resultVisualization(kinds=kinds, kind='visualization2', fps=500, maximumGapTime=0.1)
    Xa, Ya = DP_.resultData()

    for context in contexts:
        xi, yi = {}, {}
        xs, ys = {}, {}
        xa, ya = {}, {}
        #x, y = {}, {}

        title = ''
        syncname = ''
        for joint in context:
            xi[joint], yi[joint] = Xi[joint], Yi[joint]
            xa[joint], ya[joint] = Xa[joint], Ya[joint]
            title += joint + '-'
            syncname += joint + ','
        title = title[:-1]
        syncname = syncname[:-1]
        xs[syncname], ys[syncname] = Xs[joint], Ys[joint]

        resultDir = os.path.join(resultSuperDir, title)
        if not os.path.exists(resultDir):
            os.mkdir(resultDir)

        viewi = Visualization()
        viewi.show(x=xi, y=yi, xtime=refData.frame_max, ytime=inpData.frame_max,
              title=None, legend=True, correspondLine=False,
              savepath=os.path.join(resultDir, "independent.png"))

        views = Visualization()
        views.show(x=xs, y=ys, xtime=refData.frame_max, ytime=inpData.frame_max,
                  title=None, legend=True, correspondLine=False,
                  savepath=os.path.join(resultDir, "sync.png"))

        viewa = Visualization()
        viewa.show(x=xa, y=ya, xtime=refData.frame_max, ytime=inpData.frame_max,
                  title=None, legend=True, correspondLine=False,
                  savepath=os.path.join(resultDir, "async.png"))
        """
        view = Visualization()
        view.show(x=x, y=y, xtime=refData.frame_max, ytime=inpData.frame_max,
                  title=title, legend=True, correspondLine=False,
                  savepath=os.path.join(resultDir, "overlay.png"))
        """



def main():
    implementDP(dir='/Users/junkadonosuke/Desktop/research/collabo-research/keio/data/20180325/3ddata/pitcher/sekiya/',
                refname='3-1_004.csv', inpname='3-1_020.csv', refini=257, reffin=793, inpini=149, inpfin=706)

if __name__ == '__main__':
    main()