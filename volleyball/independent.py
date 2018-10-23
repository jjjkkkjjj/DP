from dp.dp import DP, Data, referenceReader
import os

def main():
    name = "IMAMURA"
    Dir = "./trc/" + name

    resultSuperDir = os.path.join('./result/', name)
    if not os.path.exists(resultSuperDir):
        os.mkdir(resultSuperDir)
    resultSuperDir = os.path.join(resultSuperDir, 'independent')
    if not os.path.exists(resultSuperDir):
        os.mkdir(resultSuperDir)

    reffile = referenceReader("{0}-normal.csv".format(name), Dir, superDir=name)
    refData = Data()
    refData.set_from_trc(os.path.join(Dir, reffile), lines='volleyball')

    for i in range(1, 7 + 1):
        filename = '{0}{1:02d}.trc'.format(name, i)

        if filename in reffile:
            continue

        inpData = Data()
        inpData.set_from_trc(os.path.join('./trc', name, filename), lines='volleyball')

        DP_ = DP(refData, inpData, verbose=False, ignoreWarning=True)

        resultDir = os.path.join(resultSuperDir, filename[:-4])
        if not os.path.exists(resultDir):
            os.mkdir(resultDir)
        DP_.calc(showresult=False, resultdir=resultDir, correspondLine=True)

    reffile = referenceReader("{0}-short.csv".format(name), Dir, superDir=name)
    refData = Data()
    refData.set_from_trc(os.path.join(Dir, reffile), lines='volleyball')

    for i in range(8, 34 + 1):
        filename = '{0}{1:02d}.trc'.format(name, i)

        if filename in reffile:
            continue

        inpData = Data()
        inpData.set_from_trc(os.path.join('./trc', name, filename), lines='volleyball')

        DP_ = DP(refData, inpData, verbose=False, ignoreWarning=True)

        resultDir = os.path.join(resultSuperDir, filename[:-4])
        if not os.path.exists(resultDir):
            os.mkdir(resultDir)
        DP_.calc(showresult=False, resultdir=resultDir)

    reffile = referenceReader("{0}-strong.csv".format(name), Dir, superDir=name)
    refData = Data()
    refData.set_from_trc(os.path.join(Dir, reffile), lines='volleyball')

    for i in range(35, 36 + 1):
        filename = '{0}{1:02d}.trc'.format(name, i)

        if filename in reffile:
            continue

        inpData = Data()
        inpData.set_from_trc(os.path.join('./trc', name, filename), lines='volleyball')

        DP_ = DP(refData, inpData, verbose=False, ignoreWarning=True)

        resultDir = os.path.join(resultSuperDir, filename[:-4])
        if not os.path.exists(resultDir):
            os.mkdir(resultDir)
        DP_.calc(showresult=False, resultdir=resultDir)


    """
    ref = Data()
    ref.set_from_trc('./trc/IMAMURA08.trc')
    inp = Data()
    inp.set_from_trc('./trc/IMAMURA34.trc')

    DP_ = DP(inp, ref)
    DP_.calc()
    """

if __name__ == '__main__':
    main()