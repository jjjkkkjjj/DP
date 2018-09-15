from DPMatching import DP, read_data
#from DPMatching_ import DP, read_data
from viewresult import ViewResult as vr
import numpy as np


input, reference = read_data('/home/junkado/Desktop/Hisamitsu/3d-data/Motion/TAURA04.CSV',
                             '/home/junkado/Desktop/Hisamitsu/3d-data/Motion/KOTOH03.CSV', dim=3, remove_rows=slice(0,6), remove_cols=slice(0,2))

"""
view = vr(input, reference)
view.view3d_input()


"""


view = vr(input, reference)
jointlist = [13,14,15,16,17]
#jointlist = [15]
#DP = [DP(input[j],reference[j], constraint='asym') for j in jointlist]
for index, j in enumerate(jointlist):
    DP_ = DP(input[jointlist[index]],reference[jointlist[index]])
    DP_.calc(jointlist[index])
    view.set_result(str(j), j, j, DP_.correspondent_point, constraint='asym')
    DP_.drawgraph(j)
    del DP_


#view.set_result("a", )
#view.view_DPResult(filepath='/home/junkado/Desktop/i-TAURA_r-koto.MP4')
#view.draw_allgraph(notall=False)
view.view_DPResult_colorbar(calculate_slope=True, forRef=False)
#DP.drawgraph()
#print DP.correspondent_point
