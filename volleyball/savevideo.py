from dp.dp import Data


for i in range(1, 7 + 1):
    data = Data()
    data.set_from_trc('./trc/IMAMURA/IMAMURA{0:02d}.trc'.format(i), lines='volleyball')
    data.save('__video/IMAMURA{0:02d}.mp4'.format(i), fps=60, saveonly=True)
exit()


for i in range(8, 34 + 1):
    data = Data()
    data.set_from_trc('./trc/IMAMURA/IMAMURA{0:02d}.trc'.format(i), lines='volleyball')
    data.save('__video/IMAMURA{0:02d}.mp4'.format(i), fps=60, saveonly=True)


for i in range(35, 36 + 1):
    data = Data()
    data.set_from_trc('./trc/IMAMURA/IMAMURA{0:02d}.trc'.format(i), lines='volleyball')
    data.save('__video/IMAMURA{0:02d}.mp4'.format(i), fps=60, saveonly=True)