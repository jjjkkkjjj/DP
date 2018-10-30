from dp.dp import DP, Data, referenceReader
from dp.view import Visualization
from dp.calccorrcoef import corrcoefMean
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

    if 'corrcoef' in method:
        corrcoef = writecorrcoef(name, initFileNum, finFileNum, showcorrcoef=False, savescvpath=os.path.join(resultSuperDir, 'correlation-coefficients.csv'))

    for i in range(initFileNum, finFileNum + 1):
        sys.stdout.write("\rcalculating now... {0}/{1}".format(i - initFileNum, finFileNum + 1 - initFileNum))
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
        if method == 'independent':
            DP_.calc(showresult=False, resultdir=resultDir, correspondLine=True)
        elif method == 'corrcoef':
            DP_.calc_corrcoef(corrcoef, showresult=False, resultdir=resultDir, correspondLine=True)
        elif method == 'comp-ind-corrcoef':
            view = Visualization()

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
        elif method == 'fixedInitial-independent':
            DP_.calcCorrespondInitial(showresult=False, resultdir=resultDir, correspondLine=True)

            view = Visualization()
            X, Y = DP_.resultData()
            view.show(x=X, y=Y, xtime=refData.frame_max, ytime=inpData.frame_max,
                      title='overlayed all matching costs', legend=True, correspondLine=True,
                      savepath=resultDir + "/overlayed-all-matching-costs.png")

        else:
            raise ValueError("{0} is invalid method".format(method))
    print("\nfinished {0}-{1} in {2}".format(serve, method, name))

def writecorrcoef(name, initFileNum, finFileNum, showcorrcoef, savescvpath):
    Datalists = []
    Dir = "./trc/" + name

    for i in range(initFileNum, finFileNum + 1):
        filename = name + '{0:02d}.trc'.format(i)
        data = Data()
        data.set_from_trc(os.path.join(Dir, filename), lines='volleyball')
        Datalists.append(data)
    return corrcoefMean(Datalists, verbose=False, showcorrcoef=showcorrcoef, savecsvpath=savescvpath)

def main(method):
    implementDP(name='IMAMURA', serve='normal', method=method, initFileNum=1, finFileNum=7)
    implementDP(name='IMAMURA', serve='short', method=method, initFileNum=8, finFileNum=34)
    implementDP(name='IMAMURA', serve='strong', method=method, initFileNum=35, finFileNum=36)


if __name__ == '__main__':
    #main(method='independent')
    #main(method='corrcoef')
    #main(method='comp-ind-corrcoef')
    main(method='fixedInitial-independent')