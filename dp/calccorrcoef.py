import numpy as np

def corrcoefMean(Datalists, verbose=False): # return Corrcoef{joint:CListneighbor} i.e Corrcoef[joint][neighbor num] = mean c
    if type(Datalists).__name__ != 'list':
        raise TypeError("Datalists must be list")
    if len(Datalists) == 0:
        print("Warning: Datalists has no element")
        return -1
    else:
        if Datalists[0].__class__.__name__ != 'Data':
            raise TypeError("Datalists\' element must be \'Data\'")


        #print("calculate correlation coefficient")

        initData = Datalists[0]
        initjointNames = list(initData.joints.keys())

        for data in Datalists:
            jointNames = list(data.joints.keys())
            if jointNames != initjointNames:
                raise ValueError("jointNames must be same among all data")

        Neighbors = {}
        ignoredJoint = []
        if verbose:
            print("The set of neighbor joint is\n\n")
        for index, joint in enumerate(initjointNames):
            # neighbor
            rows, cols = np.where(np.array(initData.lines) == index)
            cols = np.abs(cols - 1)  # revert 0,1

            neighbors = []

            for (row, col) in zip(rows, cols):
                neighbors.append(initData.lines[row][col])

            if len(neighbors) == 0:
                ignoredJoint.append(joint)
                continue
            Neighbors[joint] = np.array(neighbors)
            if verbose:
                print("{0}:{1}".format(joint, [initjointNames[i] for i in neighbors]))
            """
            head:['R_ear', 'L_ear']
            R_ear:['head', 'L_ear']
            L_ear:['head', 'R_ear']
            sternum:['R_rib', 'L_rib']
            R_rib:['sternum', 'L_rib']
            L_rib:['sternum', 'R_rib']
            R_ASIS:['L_ASIS', 'R_PSIS', 'R_frontshoulder']
            L_ASIS:['R_ASIS', 'L_PSIS', 'L_frontshoulder']
            R_PSIS:['L_PSIS', 'R_ASIS', 'R_backshoulder']
            L_PSIS:['L_ASIS', 'R_PSIS', 'L_backshoulder']
            R_frontshoulder:['R_ASIS', 'R_backshoulder', 'L_frontshoulder', 'R_in_elbow']
            R_backshoulder:['R_PSIS', 'R_frontshoulder', 'L_backshoulder', 'R_out_elbow']
            R_in_elbow:['R_frontshoulder', 'R_out_elbow', 'R_in_wrist']
            R_out_elbow:['R_backshoulder', 'R_in_elbow', 'R_out_wrist']
            R_in_wrist:['R_in_elbow', 'R_out_wrist', 'R_hand']
            R_out_wrist:['R_out_elbow', 'R_in_wrist', 'R_hand']
            R_hand:['R_in_wrist', 'R_out_wrist']
            L_frontshoulder:['L_ASIS', 'L_backshoulder', 'R_frontshoulder', 'L_in_elbow']
            L_backshoulder:['L_PSIS', 'R_backshoulder', 'L_frontshoulder', 'L_out_elbow']
            L_in_elbow:['L_frontshoulder', 'L_out_elbow', 'L_in_wrist']
            L_out_elbow:['L_backshoulder', 'L_in_elbow', 'L_out_wrist']
            L_in_wrist:['L_in_elbow', 'L_out_wrist', 'L_hand']
            L_out_wrist:['L_out_elbow', 'L_in_wrist', 'L_hand']
            L_hand:['L_in_wrist', 'L_out_wrist']
            """

        Corrcoef = {}
        if verbose:
            print("Correlation coefficients are\n\n")
        for joint in initjointNames:
            if joint in ignoredJoint:
                continue
            CList_neighbor = []
            for joint_neighbor in Neighbors[joint]:
                Cs = []
                for data in Datalists:
                    #print(data.joints[joint].shape) (time, dim)
                    #print("x:{0}".format(np.corrcoef(data.joints[joint][:, 0], data.joints[initjointNames[joint_neighbor]][:, 0])[0, 1]))
                    #print("y:{0}".format(
                    #    np.corrcoef(data.joints[joint][:, 1], data.joints[initjointNames[joint_neighbor]][:, 1])[0, 1]))
                    #print("z:{0}".format(
                    #    np.corrcoef(data.joints[joint][:, 2], data.joints[initjointNames[joint_neighbor]][:, 2])[0, 1]))
                    x_corrcoef = np.corrcoef(data.joints[joint][:, 0], data.joints[initjointNames[joint_neighbor]][:, 0])[0, 1]
                    y_corrcoef = np.corrcoef(data.joints[joint][:, 1], data.joints[initjointNames[joint_neighbor]][:, 1])[0, 1]
                    z_corrcoef = np.corrcoef(data.joints[joint][:, 2], data.joints[initjointNames[joint_neighbor]][:, 2])[0, 1]
                    Cs.append(np.mean([x_corrcoef, y_corrcoef, z_corrcoef]))
                    #print("{0}:{1}->{2}".format(joint, initjointNames[joint_neighbor], np.mean([x_corrcoef, y_corrcoef, z_corrcoef])))
                CList_neighbor.append(np.mean(np.array(Cs)))
                if verbose:
                    print("{0}:{1}->{2}".format(joint, initjointNames[joint_neighbor], CList_neighbor[-1]))
            Corrcoef[joint] = CList_neighbor
        return_dict = {'jointNames': initjointNames, 'neighbor': Neighbors, 'corrcoef': Corrcoef, 'ignored': ignoredJoint}
        return return_dict