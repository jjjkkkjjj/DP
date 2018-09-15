import os
import csv
from data_class import data
import numpy as np
import glob
from DP import DP

with open('pitch-type.csv', 'r') as f:
    reader = csv.reader(f)
    for type in reader:
        # read data
        for csvfile in type[2:]:
            tmplists = []
            with open(type[0] + csvfile, "rb") as f:
                reader = csv.reader(f)
                for row in reader:
                    tmplists.append(row)

            time = len(tmplists) - 7
            Data = data(csvfile, time, zeroisnan=True)
            Data.add_times(np.array([tmplists[i][1] for i in range(7, len(tmplists))], dtype='float'))
            index = 2
            while index < len(tmplists[6]):
                if tmplists[5][index] == 'Position':
                    if tmplists[2][index] == 'Bone Marker' and tmplists[3][index].find('Unlabeled') < 0:
                        x = np.array(
                            [tmplists[i][index] if tmplists[i][index] != '' else np.nan for i in
                             range(7, len(tmplists))],
                            dtype='float')
                        y = np.array([tmplists[i][index + 1] if tmplists[i][index + 1] != '' else np.nan for i in
                                      range(7, len(tmplists))], dtype='float')
                        z = np.array([tmplists[i][index + 2] if tmplists[i][index + 2] != '' else np.nan for i in
                                      range(7, len(tmplists))], dtype='float')
                        Data.add_joints(tmplists[3][index], x, y, z)
                    index += 3
                elif tmplists[5][index] == 'Rotation':
                    index += 4
            Data.adjust_time()
            Data.interpolate('linear')
            Data.view(savepath='__video/' + csvfile[:-4] + '.mp4', saveonly=True)


