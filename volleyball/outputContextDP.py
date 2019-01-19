from dp.contextdp import SyncContextDP, ASyncContextDP
from dp.view import Visualization
from dp.data import Data
from dp.utils import referenceReader
import os
import sys

def implementDP(name, serve, method, initFileNum, finFileNum):
    Dir = "./trc/" + name

    resultSuperDir = os.path.join('./result/', name)
    if not os.path.exists(resultSuperDir):
        os.mkdir(resultSuperDir)
    #resultSuperDir = os.path.join(resultSuperDir, 'normal-independent')
    resultSuperDir = os.path.join(resultSuperDir, '{0}-{1}'.format(serve, method))
    if not os.path.exists(resultSuperDir):
        os.mkdir(resultSuperDir)

    #reffile = referenceReader("{0}-normal.csv".format(name), Dir, superDir=name)
    reffile = referenceReader("{0}-{1}.csv".format(name, serve), Dir, superDir=name)
    refData = Data()
    refData.set_from_trc(os.path.join(Dir, reffile), lines='volleyball')
    contexts = [['head', 'R_ear', 'L_ear'],
                ['R_hand', 'R_in_wrist', 'R_out_wrist'],
                ['L_hand', 'L_in_wrist', 'L_out_wrist'],
                ['R_out_elbow', 'R_in_elbow', 'R_backshoulder'],
                ['L_out_elbow', 'L_in_elbow', 'L_backshoulder'],
                ['sternum', 'R_frontshoulder', 'L_frontshoulder'],
                ['R_rib', 'R_ASIS'],
                ['L_rib', 'L_ASIS'],
                ['R_PSIS', 'L_PSIS']]

    for i in range(initFileNum, finFileNum + 1):
        sys.stdout.write("\rcalculating now... {0}/{1}".format(i - initFileNum, finFileNum + 1 - initFileNum))
        sys.stdout.flush()

        filename = '{0}{1:02d}.trc'.format(name, i)

        if filename in reffile:
            continue

        inpData = Data()
        inpData.set_from_trc(os.path.join('./trc', name, filename), lines='volleyball')

        if 'async' in method: # async context dp
            DP_ = ASyncContextDP(contexts=contexts, reference=refData, input=inpData, verbose=False, ignoreWarning=True,
                                verboseNan=False)
        else:
            DP_ = SyncContextDP(contexts=contexts, reference=refData, input=inpData, verbose=False, ignoreWarning=True, verboseNan=False)

        resultDir = os.path.join(resultSuperDir, filename[:-4])
        if not os.path.exists(resultDir):
            os.mkdir(resultDir)
        if method == 'sync':
            DP_.synchronous()
            view = Visualization()
            X, Y = DP_.resultData()
            view.show(x=X, y=Y, xtime=refData.frame_max, ytime=inpData.frame_max,
                      title='overlayed all matching costs', legend=True, correspondLine=False,
                      savepath=resultDir + "/overlayed-all-matching-costs.png")

        elif method == 'sync-visualization':
            fps = 240
            colors = DP_.resultVisualization(fps=fps, maximumGapTime=0.1, resultDir=resultDir)
            DP_.input.show(fps=fps, colors=colors)
            DP_.input.save(path=resultDir + "/R_{0}-I_{1}.mp4".format(refData.name, inpData.name), fps=fps, colors=colors, saveonly=True)

        elif method == 'async':
            DP_.asynchronous()
            view = Visualization()
            X, Y = DP_.resultData()
            view.show(x=X, y=Y, xtime=refData.frame_max, ytime=inpData.frame_max,
                      title='overlayed all matching costs', legend=True, correspondLine=False,
                      savepath=resultDir + "/overlayed-all-matching-costs.png")

        elif method == 'async-visualization':
            fps = 240
            colors = DP_.resultVisualization(fps=fps, maximumGapTime=0.1, resultDir=resultDir)
            DP_.input.show(fps=fps, colors=colors)
            DP_.input.save(path=resultDir + "/R_{0}-I_{1}.mp4".format(refData.name, inpData.name), fps=fps, colors=colors, saveonly=True)

        else:
            raise ValueError("{0} is invalid method".format(method))
    print("\nfinished {0}-{1} in {2}".format(serve, method, name))



def main(method):
    implementDP(name='IMAMURA', serve='normal', method=method, initFileNum=1, finFileNum=7)
    implementDP(name='IMAMURA', serve='short', method=method, initFileNum=8, finFileNum=34)
    implementDP(name='IMAMURA', serve='strong', method=method, initFileNum=35, finFileNum=36)


if __name__ == '__main__':
    #main(method='sync')
    main(method='sync-visualization')