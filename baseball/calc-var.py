import os
import csv
from data_class import data
import numpy as np
import glob
from DP import DP
from view import View

def read_reference(dir, type):
    with open('reference.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == 'dir':
                if row[1] == dir and row[3] == type:
                    return row[7]
    return None


def main():
    with open('pitch-type.csv', 'r') as f:
        reader = csv.reader(f)
        for type in reader:
            # read optimal reference
            optimal_reference = read_reference(type[0], type[1])
            # read data
            DataList = []
            for csvfile in type[2:]:
                tmplists = []
                with open(type[0] + csvfile, "rb") as tf:
                    reader = csv.reader(tf)
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
                Data.interpolate('spline')
                #Data.view(savepath='/home/junkado/Desktop/mean.mp4')
                DataList.append(Data)
            # DP
            ref_data_index = type[2:].index(optimal_reference) # don't try-except
            XYZ = [] # [movement index][dim]['joint'][time]
            for index in range(len(DataList)):
                if index == ref_data_index:
                    xyz = DataList[ref_data_index].xyz()
                    joint_name = DataList[ref_data_index].joint_name
                    x = {}
                    y = {}
                    z = {}
                    for index, joint in enumerate(joint_name):
                        x[joint] = xyz[index,:,0]
                        y[joint] = xyz[index,:,1]
                        z[joint] = xyz[index,:,2]
                    XYZ.append([x,y,z])
                    continue

                dp = DP(DataList[ref_data_index], DataList[index], verbose=False)
                dp.calc(save=False)
                x, y, z = dp.input_aligned()
                XYZ.append([x, y, z])
                del dp

            # calculate mean variance
            meanx = []
            meany = []
            meanz = []
            varx = []
            vary = []
            varz = []
            Var = [] # [time][dim]
            Var.append(np.arange(DataList[ref_data_index].time))
            #mean = data('mean', DataList[ref_data_index].time)
            for index, joint in enumerate(DataList[ref_data_index].joint_name):
                x = [] # [movement index][time]
                y = []
                z = []
                for movement_index in range(len(DataList)):
                    try:
                        x.append(XYZ[movement_index][0][joint][:])
                        y.append(XYZ[movement_index][1][joint][:])
                        z.append(XYZ[movement_index][2][joint][:])
                    except KeyError:
                        continue

                meanx.append(np.mean(x, axis=0)) # mean for each time
                meany.append(np.mean(y, axis=0))
                meanz.append(np.mean(z, axis=0))
                #mean.add_joints(joint, np.mean(x, axis=0), np.mean(y, axis=0), np.mean(z, axis=0))
                #varx.append(np.var(x, axis=0)) # variance for each time [joint][time]
                #vary.append(np.var(y, axis=0))
                #varz.append(np.var(z, axis=0))
                Var.append(np.var(x, axis=0))
                Var.append(np.var(y, axis=0))
                Var.append(np.var(z, axis=0))

            Var = np.array(Var).T

            # write variance
            with open('result/variance/{0}-{1}.csv'.format(type[0].split('/')[-2], type[1]), 'w') as vf:
                vf.write('dir,{0},type,{1},\n'.format(type[0], type[1]))
                vf.write(',')
                for joint in DataList[ref_data_index].joint_name:
                    vf.write(joint + ',,,')
                vf.write('\n')
                vf.write('reference time,'+'x,y,z,'*len(DataList[ref_data_index].joint_name)+'\n')
                np.savetxt(vf, Var, delimiter=',')

            # show mean movement
            view = View()
            #view.show3d(meanx,meany,meanz)
            view.show3d(meanx, meany, meanz, fps=240, saveonly=True,
                        joint_name=DataList[ref_data_index].joint_name, savepath='__video/{0}-mean.mp4'.format(optimal_reference + type[1]))



if __name__ == '__main__':
    main()