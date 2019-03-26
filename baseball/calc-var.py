import csv
from dp.dp import DP
from dp.utils import referenceReader, csvReader
from dp.data import Data
import numpy as np
from dp.view import Visualization
from matplotlib.colors import hsv_to_rgb
import os

def main():
    with open('pitch-type.csv', 'r') as f:
        reader = csv.reader(f)
        for type in reader:
            # read data
            AlignedDataLists = [] # [csvfile index][joint index][time][dim]
            csvfiles = type[2:]
            Dir = type[0]
            name = Dir.split('/')[-2]
            # read all data and hold it to DataList
            refpath = referenceReader(name + '-' + type[1] + '.csv', Dir, superDir=name)
            #print(refpath)

            refData = csvReader(refpath, type[0])
            #refData.show()
            #exit()
            for csvfile in csvfiles:
                if csvfile == refpath:
                    continue

                inpData = csvReader(csvfile, Dir)
                DP_ = DP(refData, inpData, verbose=False, ignoreWarning=True)
                DP_.calc()
                # aligned

                alignedData = DP_.aligned()
                AlignedDataLists.append(np.array(list(alignedData.values())))
            AlignedDataLists = np.array(AlignedDataLists)
            #print(AlignedDataLists.shape) (9, 39, 1587, 3)
            # calculate mean
            mean = np.mean(AlignedDataLists, axis=0)    #(39, 1587, 3)
            meanData = Data(interpolate='linear')
            meanData.setvalues('mean movement', x=mean[:, :, 0], y=mean[:, :, 1], z=mean[:, :, 2], jointNames=list(refData.joints.keys()))

            variance = np.var(AlignedDataLists, axis=0) #(39, 1587, 3)
            std = np.std(AlignedDataLists, axis=0)
            colors = std2color(std)
            #meanData.show(colors=colors)

            # save file and bone
            meanData.save(os.path.join('result', name, '{0}-{1}-mean.MP4'.format(name, type[1])), fps=240, colors=colors, saveonly=True)

def std2color(std, maxValue=150):# maxValue=100 -> 10cm
    if not isinstance(std, np.ndarray):
        raise ValueError('variance is ndarray, but got {0}'.format(type(std).__name__))

    jnum, frame_max, dim = std.shape
    values = np.linalg.norm(std, axis=2)
    values[values > maxValue] = maxValue
    values /= maxValue  # <- normalized
    colors = []
    for j in range(jnum):
        hsv = np.zeros((frame_max, 3))
        hsv[:, 0] = 0.0
        # hsvInpArea[redIndices, 1] = scores[redIndices]
        # hsvInpArea[redIndices, 2] = 1.0
        ### center color is black
        hsv[:, 1] = 1.0
        hsv[:, 2] = values[j]

        colors.append(hsv_to_rgb(hsv))

    colors = np.array(colors).transpose((1, 0, 2))

    return colors

if __name__ == '__main__':
    main()