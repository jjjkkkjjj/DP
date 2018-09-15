# -*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
from scipy.spatial.distance import cdist


class DP:
    def __init__(self, reference, input, verbose=True):
        self.possible = True
        self.verbose = verbose
        """
        # time
        if reference.time >= input.time:
            print("reference's time must be longer than input's one")
            print("ref: {0}".format(reference.filename))
            print("inp: {0}".format(input.filename))
            self.possible = False
            return
        """
        # [joint][time][dim]
        self.input = input
        self.reference = reference

        # self.__constraint = np.array(self.__constraint)
        # [ref][input]!
        self.__local_costs = None  # x->ref,y->input
        self.__matching_costs = None  # x->ref,y->input

        self.correspondent_point = None # [reference][input]
        self.__result = {} # ['joint'][reference][input]
        self.__totalmatchingcost = None

    def __del__(self):
        if self.possible:
            del self.__local_costs
            del self.__matching_costs

    def calc(self, save=True):
        if self.verbose:
            print("reference:{0}".format(self.reference.filename))
            print("input:{0}".format(self.input.filename))
        filepath = 'result/DP_detail/' + self.reference.filename + '~' + self.input.filename + '.csv'

        data = {}

        ref_data = self.reference.xyz()
        inp_data = self.input.xyz()
        ref_data[np.isnan(ref_data)] = np.inf
        inp_data[np.isnan(inp_data)] = np.inf

        # for detection to reference
        self.__totalmatchingcost = {}

        for inp_joint_index, joint in enumerate(self.input.joint_name):
            try:
                ref_joint_index = self.reference.joint_name.index(joint)
            except ValueError:
                ref_joint_index = -1
                continue
            #if ref_joint_index == -1:
            #    continue
            if self.verbose:
                print("{0} is calculating now...".format(joint))

            # calc local cost
            self.__local_costs = cdist(ref_data[ref_joint_index,:,:], inp_data[inp_joint_index,:,:], 'euclidean')

            self.__matching_costs = np.zeros(self.__local_costs.shape)
            self.__matching_costs[0, :] = self.__local_costs[0, :]

            for reference_time in range(1, self.reference.time):
                self.__matching_costs[reference_time, 0] = self.__local_costs[reference_time, 0] + self.__matching_costs[reference_time - 1, 0]
                self.__matching_costs[reference_time, 1] = self.__local_costs[reference_time, 1] + \
                                                           np.minimum(self.__matching_costs[reference_time - 1, 0],
                                                                      self.__matching_costs[reference_time - 1, 1])
                self.__matching_costs[reference_time, 2:] = self.__local_costs[reference_time, 2:] + \
                                                           np.minimum.reduce(
                                                                      [self.__matching_costs[reference_time - 1, 2:],
                                                                      self.__matching_costs[reference_time - 1, 1:-1],
                                                                      self.__matching_costs[reference_time - 1, :-2]])
            # back track
            self.correspondent_point = []
            try:
                r, i = self.reference.time - 1, np.nanargmin(self.__matching_costs[self.reference.time - 1])
                self.correspondent_point.append([r, i])

                data_ = {}
                data_['tmc'] = self.__matching_costs[r][i]


                while r > 0 and i > 2:
                    tmp = np.argmin((self.__matching_costs[r - 1, i], self.__matching_costs[r - 1, i - 1],
                                     self.__matching_costs[r - 1, i - 2]))
                    r = r - 1
                    i = i - tmp
                    self.correspondent_point.insert(0, [r, i])

                    """
                    if tmp == 0:
                        r = r - 1
                        i = i
                    elif tmp == 1:
                        r = r - 1
                        i = i - 1
                    else:
                        r = r -1
                        i = i - 2
                    """

                while r > 0 and i > 1:
                    tmp = np.argmin((self.__matching_costs[r - 1, i], self.__matching_costs[r - 1, i - 1]))
                    r = r - 1
                    i = i - tmp
                    self.correspondent_point.insert(0, [r, i])

                while r > 0:
                    r = r - 1
                    i = 0
                    self.correspondent_point.insert(0, [r, i])

                data_['corr'] = self.correspondent_point
                data[joint] = data_

                self.__totalmatchingcost[joint] = np.nanmin(self.__matching_costs[self.reference.time - 1]) / self.reference.time

            except ValueError:
                if self.verbose:
                    print("All matching cost have nan")
                    print("skip...")
                continue

        self.__result = data

        if save:
            with open(filepath, 'w') as f:
                f.write('reference,{0},\n'.format(self.reference.filename))
                f.write("input,{0},\n".format(self.input.filename))

                row = 'joint name,'
                row2 = 'total matching cost,'
                row3 = 'reference time,'

                joints = data.keys()
                for joint in joints:
                    row += '{0},'.format(joint)
                    row2 += '{0},'.format(data[joint]['tmc'])
                    row3 += 'input time,'
                f.write(row + '\n')
                f.write(row2 + '\n')
                f.write('\n')
                f.write(row3 + '\n')

                for time in range(self.reference.time):
                    row = '{0},'.format(time)
                    for joint in joints:
                        # print(data[joint]['corr'])
                        row += '{0},'.format(data[joint]['corr'][time][1])
                    f.write(row + '\n')
            if self.verbose:
                print("saved {0}".format(filepath))

    def drawgraph(self, label="Matching Path", savefile=""):
        if not self.correspondent_point:
            print "this method must be called after .calc()"
            sys.exit()
        # setting of matplotlib
        self.__fig = plt.figure()
        plt.xlabel('reference')
        plt.ylabel('input')
        plt.xlim([0, self.input.time])
        plt.ylim([0, self.input.time])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.vlines([self.reference.time], 0, self.input.time, linestyles='dashed')
        tmp_x = np.linspace(0, self.reference.time, self.reference.time + 1)
        # plt.plot([0, self.input.time], [0, self.input.time], 'black', linestyle='dashed')
        plt.plot([0, self.reference.time],
                 [self.correspondent_point[0][1], self.reference.time + self.correspondent_point[0][1]], 'black',
                 linestyle='dashed')
        # x = np.array([self.correspondent_point[i][0] for i in range(len(self.correspondent_point))], dtype=np.int)
        # y = np.array([self.correspondent_point[i][1] for i in range(len(self.correspondent_point))], dtype=np.int)
        x = [self.correspondent_point[i][0] for i in range(len(self.correspondent_point))]
        y = [self.correspondent_point[i][1] for i in range(len(self.correspondent_point))]
        plt.plot(x, y, label=label)

        plt.legend()
        if savefile == "":
            plt.show()
        else:
            plt.savefig(savefile)

        return

    def total_matching_cost(self):
        if self.__totalmatchingcost is None:
            print("you should call calc function previously")
            return {}
        return self.__totalmatchingcost

    def input_aligned(self):
        if self.correspondent_point is None:
            print("you should call calc function previously")
            exit()

        # for each joints
        x = {}
        y = {}
        z = {}

        for joint_name, correspondent_point in self.__result.items():
            try:
                times_aligned = np.array(correspondent_point['corr'])[:,1]
                joint_index = self.input.joint_name.index(joint_name)
                x[joint_name] = self.input.x[joint_index][times_aligned]
                y[joint_name] = self.input.y[joint_index][times_aligned]
                z[joint_name] = self.input.z[joint_index][times_aligned]
            except ValueError:
                continue

        return x, y, z

    def reference_aligned(self): # x,y,z['joint'][time]
        if self.__result is None:
            print("you should call calc function previously")
            exit()

        # for each joints
        x = {}
        y = {}
        z = {}

        for joint_name, correspondent_point in self.__result.items():
            try:
                times_aligned = np.array(correspondent_point['corr'])[:,0]
                joint_index = self.input.joint_name.index(joint_name)
                x[joint_name] = self.input.x[joint_index][times_aligned]
                y[joint_name] = self.input.y[joint_index][times_aligned]
                z[joint_name] = self.input.z[joint_index][times_aligned]
            except ValueError:
                continue

        return x, y, z

