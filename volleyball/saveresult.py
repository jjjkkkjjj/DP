from DPMatching import DP, read_data
from viewresult import ViewResult as vr
import os
import numpy as np
import itertools

dir = '/home/junkado/Desktop/Hisamitsu/3d-data/Motioncsv/'
filelist = os.listdir(dir)
filecombination = list(itertools.combinations(filelist, 2))
for files in filecombination:
    input, reference = read_data(dir + files[0],
                                 dir + files[1], dim=3,
                                 remove_rows=slice(0, 6), remove_cols=slice(0, 2))

    view = vr(input, reference)
    jointlist = [13, 14, 15, 16, 17]

    for index, j in enumerate(jointlist):
        # DP = DP(input[17],reference[17])
        DP_ = DP(input[jointlist[index]], reference[jointlist[index]])
        DP_.calc(jointlist[index])
        DP_.totalcost(jointlist[index], files[0].replace('.CSV', ''),
                    '/home/junkado/Desktop/Hisamitsu/3d-data/DPresult/mds/' + files[1])
        view.set_result(str(j), j, j, DP_.correspondent_point, constraint='asym')
        del DP_


    view.view_DPResult_colorbar(calculate_slope=False, show=False, forRef=False, filepath='/home/junkado/Desktop/Hisamitsu/3d-data/DPresult/visuallize/score_I-' +
                                files[0].replace('.CSV', '-R-') + files[1].replace('.CSV', '') + '.png')
    view.view_DPResult_colorbar(calculate_slope=True, show=False, forRef=False,
                                filepath='/home/junkado/Desktop/Hisamitsu/3d-data/DPresult/visuallize/slope_I-' +
                                         files[0].replace('.CSV', '-R-') + files[1].replace('.CSV', '') + '.png')
    view.draw_allgraph(notall=True, savefile='/home/junkado/Desktop/Hisamitsu/3d-data/DPresult/graph/I-' +
                                         files[0].replace('.CSV', '-R-') + files[1].replace('.CSV', '') )

    del view
