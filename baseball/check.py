import cv2
import os
import csv
from data_class import data
import numpy as np
import glob
from DP import DP
import time as t

# read data
dir = '/home/junkado/Desktop/keio/20180325/3ddata/pitcher/sekiya/'
csvfile = '3-1_009.csv'
tmplists = []
with open(dir + csvfile, "rb") as f:
    reader = csv.reader(f)
    for row in reader:
        tmplists.append(row)

time = len(tmplists) - 7
Data = data(csvfile, time, elim_outlier=False, zeroisnan=True)
#Data = data(csvfile, time, interpolate=True)
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
Data.interpolate('spline')
#Data.elim_outlier(padding=0)
#Data.filter()
start = t.time()
Data.view(line_view=True, show_joint_name=False)
#Data.view(savepath='__video/{0}.mp4'.format(csvfile[:-4]), saveonly=True)
#print("calculation time: {0} s".format(t.time() - start))
exit()


"""
video = cv2.VideoCapture('/home/junkado/Desktop/a.MP4')

while video.isOpened():
    ret, img = video.read()

    cv2.imshow("a", img)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
"""