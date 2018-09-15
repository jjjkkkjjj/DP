import numpy as np
import matplotlib.pyplot as plt
import os
import csv

dir = '/home/junkado/Desktop/Hisamitsu/3d-data/DPresult/mds/'
filelist = os.listdir(dir)
pathlists = []
datalist = {}
nodataname = ''
for file in filelist:
    if file[-4:] == ".CSV":
        pathlists.append(dir + file)

        #data = { {} }
        with open(dir + file, "rb") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 0:
                    tmp = {"cost": -1, "ref_time": -1}
                    tmp["cost"] = float(row[2])
                    tmp["ref_time"] = int(row[3])

                    if datalist.has_key(row[1]):
                        if datalist[row[1]].has_key(file[:-4]):
                            datalist[row[1]][file[:-4]][row[0]] = tmp
                        else:
                            datalist[row[1]][file[:-4]] = {row[0]: tmp}
                    else:
                        datalist[row[1]] = {file[:-4]: {row[0]:tmp}}
                else:
                    nodatanamelist = file[:-4]

# datalist[joint_name][reference name][input name]{cost, reference time}
# !!!
#print datalist['13']

# mds data
mds = []

for joint in datalist.keys():
    plt.cla()
    # dimension of matrix
    num_data = len(datalist[joint]) + 1
    # order
    namelist = [name for name in datalist[joint].keys()]
    for name in datalist[joint].keys():
        namelist[len(datalist[joint][name]) - 1] = name
    namelist.insert(0, nodatanamelist)
    #print namelist
    # matrix
    Distance = np.zeros((num_data, num_data))

    for i, reference_name in enumerate(namelist):
        if i != 0:
            for j, input_name in enumerate(datalist[joint][reference_name]):
                tmp = datalist[joint][reference_name][namelist[j]]
                Distance[i][j] = tmp['cost']/tmp['ref_time']
    Distance = Distance + Distance.T

    N = len(Distance)
    S = Distance * Distance

    one = np.eye(N) - np.ones((N, N)) / N

    P = - 1.0 / 2 * one * S * one

    w, v = np.linalg.eig(P)
    ind = np.argsort(w)
    x1 = ind[-1]
    x2 = ind[-2]
    print w[x1], w[x2]

    s = P.std(axis=0)
    w1 = s[x1]
    w2 = s[x2]

    mds_tmp = []
    for i in range(N):
        mds_tmp.append([w1 * v[i, x1], w2 * v[i, x2]])
        plt.plot(w1 * v[i, x1], w2 * v[i, x2], 'b.')
        plt.annotate(namelist[i] + "-" + joint, (w1 * v[i, x1], w2 * v[i, x2]))
    mds.append(mds_tmp)
    plt.draw()
    #plt.show()
    plt.savefig(dir + joint + ".png")

plt.cla()
for i, joint in enumerate(datalist.keys()):
    for j in range(N):
        plt.plot(mds[i][j][0], mds[i][j][1], 'b.')
        plt.annotate(namelist[j] + "-" + joint, (mds[i][j][0], mds[i][j][1]))

plt.savefig(dir + "-all.png")