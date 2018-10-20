import csv
from dp.dp import DP, referenceReader, Data
import numpy as np
from dp.view import Visualization
from dp.calccorrcoef import corrcoefMean

def main():
    corrcoef = writecorrcoef()

    reffile = referenceReader("IMAMURA-normal.csv", "./trc/IMAMURA")
    refData = Data()
    refData.set_from_trc('./trc/{0}'.format(reffile), lines='volleyball')
    for i in range(1, 7 + 1):
        if 'IMAMURA{0:02d}.trc'.format(i) in reffile:
            continue
        inpData = Data()
        inpData.set_from_trc('./trc/IMAMURA/IMAMURA{0:02d}.trc'.format(i), lines='volleyball')

        DP_ = DP(refData, inpData, verbose=False, ignoreWarning=True)
        DP_.calc_corrcoef(corrcoef, showresult=False, resultdir='./result/IMAMURA-normal-corrcoef-ver/')
        exit()
def writecorrcoef():
    Datalists = []
    for i in range(1, 7 + 1):
        data = Data()
        data.set_from_trc('./trc/IMAMURA/IMAMURA{0:02d}.trc'.format(i), lines='volleyball')
        Datalists.append(data)
    return corrcoefMean(Datalists, verbose=False)

if __name__ == '__main__':
    main()
    #writecorrcoef()