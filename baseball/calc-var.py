import csv
from csvReader import csvReader
from dp.dp import DP, referenceReader, Data
import numpy as np
from dp.view import Visualization

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

            meanData.show()

            print(mean.shape)
            print(variance.shape)
            exit()
            # save file and bone

if __name__ == '__main__':
    main()