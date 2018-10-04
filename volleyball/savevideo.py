from dp.dp import DP, Data, referenceDetector

Datalists = []
for i in range(1, 7 + 1):
    data = Data()
    data.set_from_trc('./trc/IMAMURA{0:02d}.trc'.format(i), lines='volleyball')
    Datalists.append(data)
referenceDetector(Datalists, 'IMAMURA-normal.csv')

Datalists = []
for i in range(8, 34 + 1):
    data = Data()
    data.set_from_trc('./trc/IMAMURA{0:02d}.trc'.format(i), lines='volleyball')
    Datalists.append(data)
referenceDetector(Datalists, 'IMAMURA-strong.csv')

Datalists = []
for i in range(35, 36 + 1):
    data = Data()
    data.set_from_trc('./trc/IMAMURA{0:02d}.trc'.format(i), lines='volleyball')
    Datalists.append(data)
referenceDetector(Datalists, 'IMAMURA-short.csv')