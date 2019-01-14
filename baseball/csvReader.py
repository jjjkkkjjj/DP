import csv
from dp.data import Data
import numpy as np


def csvReader(csvfile, dir):

    tmplists = []

    with open(dir + csvfile, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            tmplists.append(row)

    time = len(tmplists) - 7
    data = Data(interpolate='linear')

    jointNames = []
    X = []
    Y = []
    Z = []

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
                X.append(x)
                Y.append(y)
                Z.append(z)
                jointNames.append(tmplists[3][index])
            index += 3
        elif tmplists[5][index] == 'Rotation':
            index += 4

    data.setvalues(csvfile, np.array(X), np.array(Y), np.array(Z), jointNames, dir=dir, lines='baseball')

    return data