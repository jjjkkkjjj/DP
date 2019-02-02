from dp.contextdp import SyncContextDP, AsyncContextDP
from dp.dp import DP
from dp.view import Visualization
from dp.data import Data
from dp.utils import referenceReader
import os
import sys

def implementDP(name, serve, successNum, failureNum):
    Dir = "./trc/" + name

    resultSuperDir = os.path.join('./compareDP/', name)
    if not os.path.exists(resultSuperDir):
        os.mkdir(resultSuperDir)
    #resultSuperDir = os.path.join(resultSuperDir, 'normal-independent')
    resultSuperDir = os.path.join(resultSuperDir, '{0}-s{1}-f{2}'.format(serve, successNum, failureNum))
    if not os.path.exists(resultSuperDir):
        os.mkdir(resultSuperDir)

    #reffile = referenceReader("{0}-normal.csv".format(name), Dir, superDir=name)
    filename = '{0}{1:02d}.trc'.format(name, successNum)
    refData = Data()
    refData.set_from_trc(os.path.join('./trc', name, filename), lines='volleyball')

    filename = '{0}{1:02d}.trc'.format(name, failureNum)
    inpData = Data()
    inpData.set_from_trc(os.path.join('./trc', name, filename), lines='volleyball')

    contexts = [['head', 'R_ear', 'L_ear'],
                ['R_hand', 'R_in_wrist', 'R_out_wrist'],
                ['L_hand', 'L_in_wrist', 'L_out_wrist'],
                ['R_out_elbow', 'R_in_elbow', 'R_backshoulder'],
                ['L_out_elbow', 'L_in_elbow', 'L_backshoulder'],
                ['sternum', 'R_frontshoulder', 'L_frontshoulder'],
                ['R_rib', 'R_ASIS'],
                ['L_rib', 'L_ASIS'],
                ['R_PSIS', 'L_PSIS']]

    kinds = ['async3-visualization2',
            'async3-visualization2',
            'async3-visualization2',
            'async3-visualization2',
            'async3-visualization2',
            'async3-visualization2',
            'async2-visualization2',
            'async2-visualization2',
            'async2-visualization2']


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
        for joint in context:
            xi[joint], yi[joint] = Xi[joint], Yi[joint]
            xs[joint], ys[joint] = Xs[joint], Ys[joint]
            xa[joint], ya[joint] = Xa[joint], Ya[joint]
            title += joint + '-'
        title = title[:-1]

        resultDir = os.path.join(resultSuperDir, title)
        if not os.path.exists(resultDir):
            os.mkdir(resultDir)

        view = Visualization()
        view.show(x=xi, y=yi, xtime=refData.frame_max, ytime=inpData.frame_max,
              title=title, legend=True, correspondLine=False,
              savepath=os.path.join(resultDir, "independent.png"))

        view = Visualization()
        view.show(x=xs, y=ys, xtime=refData.frame_max, ytime=inpData.frame_max,
                  title=title, legend=True, correspondLine=False,
                  savepath=os.path.join(resultDir, "sync.png"))

        view = Visualization()
        view.show(x=xa, y=ya, xtime=refData.frame_max, ytime=inpData.frame_max,
                  title=title, legend=True, correspondLine=False,
                  savepath=os.path.join(resultDir, "async.png"))
        """
        view = Visualization()
        view.show(x=x, y=y, xtime=refData.frame_max, ytime=inpData.frame_max,
                  title=title, legend=True, correspondLine=False,
                  savepath=os.path.join(resultDir, "overlay.png"))
        """



def main():
    implementDP(name='IMAMURA', serve='normal', successNum=7, failureNum=1)
    implementDP(name='IMAMURA', serve='short', successNum=8, failureNum=34)
    implementDP(name='IMAMURA', serve='strong', successNum=36, failureNum=35)


if __name__ == '__main__':
    main()