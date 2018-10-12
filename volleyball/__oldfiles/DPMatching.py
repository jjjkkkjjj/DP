import cv2
import numpy as np
import sys
import math
import csv
import matplotlib.pyplot as plt


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
            #input_data.append(row)

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
            #reference_data.append(row)
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
    if input_data.shape[1]%dim != 0 or reference_data.shape[1]%dim != 0:
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


    for data_index in range(0, input_data.shape[1], dim) :
        tmp_row = []
        for time in range(0, input_data.shape[0]):
            tmp = []
            for i in range(dim):
                tmp.append(input_data[time][i + data_index])
            tmp_row.append(tmp)
        return_inp_data.append(tmp_row)

    return np.array(return_inp_data), np.array(return_ref_data)

class DP:

    def __init__(self, input, reference, constraint='default', alltime=True, local_func_change=False):
        if input.shape[1] != reference.shape[1]:
            print "The column number of both input and refenrence must be same"
            print "row:time,col:dimension"
            self.__init = False
            return
        if input.shape[1] < reference.shape[1]:
            print "the time length of input must be longer than reference's"
            self.__init = False
            return

        self.__input = input
        self.__input_time = input.shape[0]
        self.__data_dimension = input.shape[1]
        self.__reference = reference
        self.__reference_time = reference.shape[0]
        
        self.__set_constraint(constraint)
        #self.__constraint = np.array(self.__constraint)
        #[ref][input]!
        self.__local_costs = np.zeros((self.__reference_time, self.__input_time))#x->ref,y->input
        self.__matching_costs = np.zeros((self.__reference_time, self.__input_time)) # x->ref,y->input
        self.__back_track_ref = np.ones((self.__reference_time, self.__input_time)) * -1
        self.__back_track_inp = np.ones((self.__reference_time, self.__input_time)) * -1
        if not alltime:
            for t_input in range(1, self.__input_time):
                self.__local_costs[0][t_input] = np.inf
        self.__func_change = local_func_change
        self.correspondent_point = []

        self.__totalmatchingcost = {}
        self.__init = True

    def __del__(self):
        if self.__init:
            del self.__local_costs
            del self.__matching_costs
            del self.__back_track_ref
            del self.__back_track_inp

    def __set_constraint(self, constraint):
        if constraint == 'default':
            self.__add_matching_costs = [[-1, 0, 1], [-1, -1, 1], [-1, -2, 1]]
            self.__add_local_costs = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]#[ref1,inp1,coefficient1, ref2,...]
            self.__back_tracks = [[-1, 0], [-1, -1], [-1, -2]]#[ref1,inp1,ref2,inp2,...] # these three arguments' length must be same
            #self.__constraint = [[[-1, 0]], [[-1, -1]], [[-1, -2]]]
        elif constraint == 'asym':
            self.__add_matching_costs = [[-1, 0, 1], [-1, -1, 1], [-1, -2, 1]]
            self.__add_local_costs = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]#[ref1,inp1,coefficient1, ref2,...]
            self.__back_tracks = [[-1, 0], [-1, -1], [-1, -2]]#[ref1,inp1,ref2,inp2,...] # these three arguments' length must be same
            #self.__constraint = [[[-1, 0]], [[-1, -1]], [[-1, -2]]]
        elif constraint == 'sym':
            self.__add_matching_costs = [[-1, 0, 1], [-1, -1, 1], [0, -1, 1]]
            self.__add_local_costs = [[0, 0, 1], [0, 0, 2], [0, 0, 1]]#[ref1,inp1,coefficient1, ref2,...]
            self.__back_tracks = [[-1, 0], [-1, -1], [0, -1]]#[ref1,inp1,ref2,inp2,...] # these three arguments' length must be same
            #self.__constraint = [[[-1, 0]], [[-1, -1]], [[0, -1]]]
            self.__add_matching_costs = [[-2, -1, 1], [-1, -1, 1], [-1, -2, 1]]
            self.__add_local_costs = [[0,0,1, -1,0,2], [0, 0, 2], [0,0,1, 0,-1,2]]#[ref1,inp1,coefficient1, ref2,...]
            self.__back_tracks = [[-1,0, -2,-1], [-1, -1], [0,-1, -1,-2]]#[ref1,inp1,ref2,inp2,...] # these three arguments' length must be same
            #self.__constraint = [[[-1, 0], [-2, -1]], [[-1, -1]], [[0, -1], [-1, -2]]]
        elif constraint == 'skip1':
            self.__add_matching_costs = [[-2, -1, 1], [-1, -1, 1], [-1, -2, 1]]
            self.__add_local_costs = [[0,0,1, -1,0,1], [0, 0, 1], [0,0,1]]#[ref1,inp1,coefficient1, ref2,...]
            self.__back_tracks = [[-1,0, -2,-1], [-1, -1], [-1,-2]]#[ref1,inp1,ref2,inp2,...] # these three arguments' length must be same
        elif constraint == 'skip2':
            self.__add_matching_costs = [[-2, -1, 1], [-1, -1, 1], [-1, -2, 1]]
            self.__add_local_costs = [[0,0,1], [0, 0, 1], [0,0,1, 0,-1,1]]#[ref1,inp1,coefficient1, ref2,...]
            self.__back_tracks = [[-2,-1], [-1, -1], [0,-1, -1,-2]]#[ref1,inp1,ref2,inp2,...] # these three arguments' length must be same
        else:
            self.__add_matching_costs = [[-1, 0, 1], [-1, -1, 1], [-1, -2, 1]]
            self.__add_local_costs = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]#[ref1,inp1,coefficient1, ref2,...]
            self.__back_tracks = [[-1, 0], [-1, -1], [-1, -2]]#[ref1,inp1,ref2,inp2,...] # these three arguments' length must be same
        return

    """
    elif constraint == 'symlong'
        constraint =[[[-1, 0], [-2, -1]], [-1, -1], [[0, -1], [-1, -2]]]
    """

    def set_func_local(self, Class, func):
        if self.__func_change:
            self.__func_name = func.__name__
            setattr(Class, self.__func_name, func)
        else:
            print ".set_func_local method was pointless"

    def __func_call(self, s, func_name, args=[]):
        getattr(s, func_name)(*args)

    def __default_func_local(self, input_data, reference_data):
        local_cost = 0
        for dim in range(self.__data_dimension):
            local_cost += (input_data[dim] - reference_data[dim])**2
        return math.sqrt(local_cost)

    def calc(self, joint_index):
        if not self.__init:
            return False
        # local cost
        for t_reference in range(self.__reference_time):
            for t_input in range(self.__input_time):
                if self.__func_change:
                    self.__local_costs[t_reference][t_input] += \
                        self.__func_call(self, self.__func_name, [self.__input[t_input], self.__reference[t_reference]])
                else:
                    self.__local_costs[t_reference][t_input] += self.__default_func_local(self.__input[t_input], self.__reference[t_reference])

        for t_input in range(self.__input_time):
            self.__matching_costs[0][t_input] = self.__local_costs[0][t_input]

        # matching cost
        for t_reference in range(1, self.__reference_time):
            for t_input in range(self.__input_time):
                tmp = np.zeros(len(self.__add_matching_costs))
                for cons_index in range(len(self.__add_matching_costs)):
                    ref_time = t_reference + self.__add_matching_costs[cons_index][0]
                    inp_time = t_input + self.__add_matching_costs[cons_index][1]
                    if ref_time >= 0 and ref_time < self.__reference_time and inp_time >= 0 and inp_time < self.__input_time:
                        tmp[cons_index] += self.__matching_costs[ref_time][inp_time]*self.__add_matching_costs[cons_index][2]
                    else:
                        tmp[cons_index] = np.nan
                        continue
                    for cons_m_index in range(0, len(self.__add_local_costs[cons_index]), 3):
                        ref_time = t_reference + self.__add_local_costs[cons_index][cons_m_index]
                        inp_time = t_input + self.__add_local_costs[cons_index][cons_m_index + 1]
                        if ref_time >= 0 and ref_time < self.__reference_time and inp_time >= 0 and inp_time < self.__input_time:
                            tmp[cons_index] += self.__local_costs[ref_time][inp_time]*self.__add_local_costs[cons_index][cons_m_index + 2]
                        else:
                            tmp[cons_index] = np.nan
                            break

                try:
                    self.__matching_costs[t_reference][t_input] = np.nanmin(tmp)
                    tmp_ref_time = t_reference
                    tmp_inp_time = t_input
                    argmentmin = np.nanargmin(tmp)
                    for backtrack_num in range(0, len(self.__back_tracks[argmentmin]), 2):
                        self.__back_track_ref[tmp_ref_time][tmp_inp_time] = int(t_reference + self.__back_tracks[argmentmin][backtrack_num])
                        self.__back_track_inp[tmp_ref_time][tmp_inp_time] = int(t_input + self.__back_tracks[argmentmin][backtrack_num + 1])
                        ttmp_ref_time = tmp_ref_time
                        tmp_ref_time = int(self.__back_track_ref[tmp_ref_time][tmp_inp_time])
                        tmp_inp_time = int(self.__back_track_inp[ttmp_ref_time][tmp_inp_time])

                except ValueError:
                    self.__matching_costs[t_reference][t_input] = np.inf
                    self.__back_track_ref[t_reference][t_input] = -1
                    self.__back_track_inp[t_reference][t_input] = -1

                # backtrack

        tmp_ref_time = self.__reference_time - 1
        tmp_inp_time = np.argmin(self.__matching_costs[tmp_ref_time])

        self.__totalmatchingcost[str(joint_index)] = self.__matching_costs[tmp_ref_time][tmp_inp_time]

        while tmp_ref_time >= 0:
            self.correspondent_point.append([tmp_ref_time, tmp_inp_time])
            ref_now = tmp_ref_time
            tmp_ref_time = int(self.__back_track_ref[tmp_ref_time][tmp_inp_time])
            tmp_inp_time = int(self.__back_track_inp[ref_now][tmp_inp_time])

        self.correspondent_point.reverse()
	
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
        #plt.plot([0, self.__input_time], [0, self.__input_time], 'black', linestyle='dashed')
        plt.plot([0, self.__reference_time], [self.correspondent_point[0][1], self.__reference_time + self.correspondent_point[0][1]], 'black', linestyle='dashed')
        #x = np.array([self.correspondent_point[i][0] for i in range(len(self.correspondent_point))], dtype=np.int)
        #y = np.array([self.correspondent_point[i][1] for i in range(len(self.correspondent_point))], dtype=np.int)
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
            writer.writerow([input_name, joint_index, self.__totalmatchingcost[str(joint_index)], self.__reference_time])
        return
