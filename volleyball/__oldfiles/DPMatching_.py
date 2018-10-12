import cv2
import numpy as np
import sys
import math
import csv
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def read_data(input_filepath, reference_filepath, dim, remove_rows=None, remove_cols=None):
    input_data = []
    reference_data = []

    with open(input_filepath, "rb") as f:
        reader = csv.reader(f)
        for row in reader:
            if remove_cols is not None:
                del row[remove_cols]
            if not row:
                continue
            # input_data.append(row)

            tmp = []
            for data in row:
                if not data:
                    tmp.append('nan')
                    continue
                tmp.append(data)
            input_data.append(tmp)

    with open(reference_filepath, "rb") as f:
        reader = csv.reader(f)
        for row in reader:
            if remove_cols is not None:
                del row[remove_cols]
            if not row:
                continue
            # reference_data.append(row)
            tmp = []
            for data in row:
                if not data:
                    tmp.append('nan')
                    continue
                tmp.append(data)
            reference_data.append(tmp)

    if remove_rows is not None:
        del input_data[remove_rows]
        del reference_data[remove_rows]

    input_data = np.array(input_data, dtype=np.float)
    reference_data = np.array(reference_data, dtype=np.float)
    if input_data.shape[1] % dim != 0 or reference_data.shape[1] % dim != 0:
        print "dimension error"

    return_ref_data = []
    return_inp_data = []
    for data_index in range(0, reference_data.shape[1], dim):
        tmp_row = []
        for time in range(0, reference_data.shape[0]):
            tmp = []
            for i in range(dim):
                tmp.append(reference_data[time][i + data_index])
            tmp_row.append(tmp)
        return_ref_data.append(tmp_row)

    for data_index in range(0, input_data.shape[1], dim):
        tmp_row = []
        for time in range(0, input_data.shape[0]):
            tmp = []
            for i in range(dim):
                tmp.append(input_data[time][i + data_index])
            tmp_row.append(tmp)
        return_inp_data.append(tmp_row)

    return np.array(return_inp_data), np.array(return_ref_data)


class DP:

    def __init__(self, input, reference):
        if input.shape[1] != reference.shape[1]:
            print "The column number of both input and refenrence must be same"
            print "row:time,col:dimension"
            return
        if input.shape[1] < reference.shape[1]:
            print "the time length of input must be longer than reference's"
            return

        self.__input = input
        self.__input_time = input.shape[0]
        self.__data_dimension = input.shape[1]
        self.__reference = reference
        self.__reference_time = reference.shape[0]

        # self.__constraint = np.array(self.__constraint)
        # [ref][input]!
        self.__local_costs = None  # x->ref,y->input
        self.__matching_costs = None  # x->ref,y->input

        self.correspondent_point = []

        self.__totalmatchingcost = {}

    def __del__(self):
        del self.__local_costs
        del self.__matching_costs

    def __set_constraint(self, constraint):
        if constraint == 'default':
            self.__add_matching_costs = [[-1, 0, 1], [-1, -1, 1], [-1, -2, 1]]
            self.__add_local_costs = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]  # [ref1,inp1,coefficient1, ref2,...]
            self.__back_tracks = [[-1, 0], [-1, -1],
                                  [-1, -2]]  # [ref1,inp1,ref2,inp2,...] # these three arguments' length must be same
            # self.__constraint = [[[-1, 0]], [[-1, -1]], [[-1, -2]]]
        elif constraint == 'asym':
            self.__add_matching_costs = [[-1, 0, 1], [-1, -1, 1], [-1, -2, 1]]
            self.__add_local_costs = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]  # [ref1,inp1,coefficient1, ref2,...]
            self.__back_tracks = [[-1, 0], [-1, -1],
                                  [-1, -2]]  # [ref1,inp1,ref2,inp2,...] # these three arguments' length must be same
            # self.__constraint = [[[-1, 0]], [[-1, -1]], [[-1, -2]]]
        elif constraint == 'sym':
            self.__add_matching_costs = [[-1, 0, 1], [-1, -1, 1], [0, -1, 1]]
            self.__add_local_costs = [[0, 0, 1], [0, 0, 2], [0, 0, 1]]  # [ref1,inp1,coefficient1, ref2,...]
            self.__back_tracks = [[-1, 0], [-1, -1],
                                  [0, -1]]  # [ref1,inp1,ref2,inp2,...] # these three arguments' length must be same
            # self.__constraint = [[[-1, 0]], [[-1, -1]], [[0, -1]]]
        elif constraint == 'symlong':
            self.__add_matching_costs = [[-2, -1, 1], [-1, -1, 1], [-1, -2, 1]]
            self.__add_local_costs = [[0, 0, 1, -1, 0, 2], [0, 0, 2],
                                      [0, 0, 1, 0, -1, 2]]  # [ref1,inp1,coefficient1, ref2,...]
            self.__back_tracks = [[-1, 0, -2, -1], [-1, -1], [0, -1, -1,
                                                              -2]]  # [ref1,inp1,ref2,inp2,...] # these three arguments' length must be same
            # self.__constraint = [[[-1, 0], [-2, -1]], [[-1, -1]], [[0, -1], [-1, -2]]]
        elif constraint == 'skip1':
            self.__add_matching_costs = [[-2, -1, 1], [-1, -1, 1], [-1, -2, 1]]
            self.__add_local_costs = [[0, 0, 1, -1, 0, 1], [0, 0, 1], [0, 0, 1]]  # [ref1,inp1,coefficient1, ref2,...]
            self.__back_tracks = [[-1, 0, -2, -1], [-1, -1],
                                  [-1, -2]]  # [ref1,inp1,ref2,inp2,...] # these three arguments' length must be same
        elif constraint == 'skip2':
            self.__add_matching_costs = [[-2, -1, 1], [-1, -1, 1], [-1, -2, 1]]
            self.__add_local_costs = [[0, 0, 1], [0, 0, 1], [0, 0, 1, 0, -1, 1]]  # [ref1,inp1,coefficient1, ref2,...]
            self.__back_tracks = [[-2, -1], [-1, -1], [0, -1, -1,
                                                       -2]]  # [ref1,inp1,ref2,inp2,...] # these three arguments' length must be same
        else:
            self.__add_matching_costs = [[-1, 0, 1], [-1, -1, 1], [-1, -2, 1]]
            self.__add_local_costs = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]  # [ref1,inp1,coefficient1, ref2,...]
            self.__back_tracks = [[-1, 0], [-1, -1],
                                  [-1, -2]]  # [ref1,inp1,ref2,inp2,...] # these three arguments' length must be same
        return

    """
    elif constraint == 'symlong'
        constraint =[[[-1, 0], [-2, -1]], [-1, -1], [[0, -1], [-1, -2]]]
    """

    def __default_func_local(self, input_data, reference_data):
        local_cost = 0
        for dim in range(self.__data_dimension):
            local_cost += (input_data[dim] - reference_data[dim]) ** 2
        return math.sqrt(local_cost)

    def calc(self, joint_index, alltime=True, constraint='default', local_dist='euclidean'):
        self.__set_constraint(constraint)

        # local
        self.local_dist = local_dist
        self.__local_costs = cdist(self.__reference, self.__input, self.local_dist)

        if not alltime:
            self.__local_costs[0,:] = np.inf

        self.__matching_costs = np.zeros(self.__local_costs.shape)
        self.__matching_costs[0, :] = self.__local_costs[0, :]

        # matching cost
        for reference_time in range(1, self.__reference_time):
            self.__matching_costs[reference_time, 0] = self.__local_costs[reference_time, 0] + self.__matching_costs[
                reference_time - 1, 0]
            self.__matching_costs[reference_time, 1] = self.__local_costs[reference_time, 1] + \
                                                       np.minimum(self.__matching_costs[reference_time - 1, 0],
                                                                  self.__matching_costs[reference_time - 1, 1])
            self.__matching_costs[reference_time, 2:] = self.__local_costs[reference_time, 2:] + \
                                                        np.minimum.reduce(
                                                            [self.__matching_costs[reference_time - 1, 2:],
                                                             self.__matching_costs[reference_time - 1, 1:-1],
                                                             self.__matching_costs[reference_time - 1, :-2]])

        # backtrack
        r, i = self.__reference_time - 1, np.argmin(self.__matching_costs[self.__reference_time - 1])

        self.correspondent_point.append([r, i])
        while r > 0 and i > 0:
            tmp = np.argmin(
                (self.__matching_costs[r - 1, i], self.__matching_costs[r - 1, i - 1], self.__matching_costs[r - 1, i - 2]))
            r = r - 1
            i = i - tmp
            self.correspondent_point.insert(0, [r, i])

        return

    def drawgraph(self, label="Matching Path", savefile=""):
        if not self.correspondent_point:
            print "this method must be called after .calc()"
            sys.exit()
        # setting of matplotlib
        self.__fig = plt.figure()
        plt.xlabel('reference')
        plt.ylabel('input')
        plt.xlim([0, self.__input_time])
        plt.ylim([0, self.__input_time])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.vlines([self.__reference_time], 0, self.__input_time, linestyles='dashed')
        tmp_x = np.linspace(0, self.__reference_time, self.__reference_time + 1)
        # plt.plot([0, self.__input_time], [0, self.__input_time], 'black', linestyle='dashed')
        plt.plot([0, self.__reference_time],
                 [self.correspondent_point[0][1], self.__reference_time + self.correspondent_point[0][1]], 'black',
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

    def totalcost(self, joint_index, input_name, filepath):
        if not self.correspondent_point:
            print "this method must be called after .calc()"
            sys.exit()

        with open(filepath, 'a') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(
                [input_name, joint_index, self.__totalmatchingcost[str(joint_index)], self.__reference_time])
        return
