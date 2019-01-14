from dp.dp import DP
from dp.data import Data
from dp.utils import referenceDetector

Datalists = []
for i in range(1, 7 + 1):
    data = Data()
    data.set_from_trc('./trc/IMAMURA/IMAMURA{0:02d}.trc'.format(i), lines='volleyball')
    Datalists.append(data)
referenceDetector(Datalists, 'IMAMURA-normal.csv', superDir='IMAMURA', verbose=False, verboseNan=False)

Datalists = []
for i in range(8, 34 + 1):
    data = Data()
    data.set_from_trc('./trc/IMAMURA/IMAMURA{0:02d}.trc'.format(i), lines='volleyball')
    Datalists.append(data)
referenceDetector(Datalists, 'IMAMURA-strong.csv', superDir='IMAMURA', verbose=False, verboseNan=False)

Datalists = []
for i in range(35, 36 + 1):
    data = Data()
    data.set_from_trc('./trc/IMAMURA/IMAMURA{0:02d}.trc'.format(i), lines='volleyball')
    Datalists.append(data)
referenceDetector(Datalists, 'IMAMURA-short.csv', superDir='IMAMURA', verbose=False, verboseNan=False)
"""
ref = Data()
ref.set_from_trc('./trc/IMAMURA08.trc')
inp = Data()
inp.set_from_trc('./trc/IMAMURA34.trc')

DP_ = DP(inp, ref)
DP_.calc()
"""