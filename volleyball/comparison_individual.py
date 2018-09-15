from dp.dp import DP, Data

Input = []
Reference = None
for i in range(1, 7 + 1):
    data = Data()
    data.set_from_trc('./trc/IMAMURA{0:02d}.trc'.format(i))
    if i == 5:
        Reference = data
        continue
    Input.append(data)
for Inp in Input:
    DP_ = DP(Reference, Inp, verbose=False, ignoreWarning=True)
    DP_.calc(resultdir='./result/IMAMURA-normal')
"""
Input = []
Reference = None
for i in range(8, 34 + 1):
    data = Data()
    data.set_from_trc('./trc/IMAMURA{0:02d}.trc'.format(i))
    if i == 19:
        Reference = data
        continue
    Input.append(data)
for Inp in Input:
    DP_ = DP(Reference, Inp, verbose=False, ignoreWarning=True)
    DP_.calc(resultdir='./result/IMAMURA-short')

for i in range(35, 36 + 1):
    data = Data()
    data.set_from_trc('./trc/IMAMURA{0:02d}.trc'.format(i))
    if i == 35:
        Reference = data
        continue
    Input.append(data)
for Inp in Input:
    DP_ = DP(Reference, Inp, verbose=False, ignoreWarning=True)
    DP_.calc(resultdir='./result/IMAMURA-strong')
"""
"""
ref = Data()
ref.set_from_trc('./trc/IMAMURA08.trc')
inp = Data()
inp.set_from_trc('./trc/IMAMURA34.trc')

DP_ = DP(inp, ref)
DP_.calc()
"""