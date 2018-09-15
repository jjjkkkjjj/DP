import numpy as np
from scipy.sparse.linalg import eigs

def PCA_based_reconstruction(x, y, z):
    x = np.array(x) # [joint][time]
    y = np.array(y)
    z = np.array(z)

    # make a Matrix M [time][joint]
    # matrix with marker data organized in the form
    # [x1(t1), y1(t1), z1, x2, y2, z2, x3, ..., zn(t1)
    # x1(t2), y1(t2), ...
    # ......
    # x1(tm), y1(tm), ..., zn(tm)]
    mean_x = np.mean(x[~np.isnan(x).any(axis=1), :], axis=0)
    mean_y = np.mean(y[~np.isnan(y).any(axis=1), :], axis=0)
    mean_z = np.mean(z[~np.isnan(z).any(axis=1), :], axis=0)

    M = np.vstack((x[0] - mean_x, y[0] - mean_y, z[0] - mean_z))
    for joint in range(1, x.shape[0]):
        M = np.vstack((M, x[joint] - mean_x, y[joint] - mean_y, z[joint] - mean_z))

    # define some matrix
    M = M.T
        # make a Matrix has complete marker
    row_with_nan, column_with_nan = np.where(np.isnan(M))

    M_zeros = M.copy()
    M_zeros[:, column_with_nan] = 0

    N = M[~np.isnan(M).any(axis=1), :]

    N_zeros = N.copy()
    N_zeros[:, column_with_nan] = 0

    # for normalization
    mean_N = np.mean(N, axis=0)[np.newaxis, :]
    mean_N_zeros = np.mean(N_zeros, axis=0)[np.newaxis, :]
    std_N = np.std(N, axis=0, ddof=0)[np.newaxis, :]

    # normalization
    M_zeros = (M_zeros - mean_N_zeros) / std_N # * weights for now not use weights

    N = (N - mean_N) / std_N
    N_zeros = (N_zeros - mean_N_zeros) / std_N

    # PCA for N
    PC_values, PC_vecs = PCA(N)

    # PCA for N_zeros
    PC_zeros_values, PC_zeros_vecs = PCA(N_zeros)

    # T is got by multiplying PC_zeros_vecs and PC_vecs
    T = np.dot(PC_vecs.T, PC_zeros_vecs)

    # main
    #R = M_zeros.dot(PC_zeros_vecs).dot(T).dot(PC_vecs.T)
    R = np.dot(M_zeros, PC_zeros_vecs)
    R = np.dot(R, T)
    R = np.dot(R, PC_vecs.T)

    # reverse normalization
    R = (mean_N + R*std_N)
    """
    for joint in range(x.shape[0]):
        R[:, 3*joint + 0] += mean_x[joint]
        R[:, 3 * joint + 1] += mean_y[joint]
        R[:, 3 * joint + 2] += mean_z[joint]
    """

    R[:, 0::3] += mean_x[np.newaxis, :].T
    R[:, 1::3] += mean_y[np.newaxis, :].T
    R[:, 2::3] += mean_z[np.newaxis, :].T
    R = R.real
    #print(x[0])
    #print(R[:,0])
    #print()
    return R[:, 0::3].T, R[:, 2::3].T, R[:, 1::3].T

def PCA(Data):
    eigs_num = min(40, Data.shape[1] - 3)
    start_vec = np.ones((Data.shape[1], 1))
    #eigs_num=Data.shape[1]
    c = np.dot(Data.T, Data) / Data.shape[0] # Data has already substracted mean

    values, vecs = eigs(c, k=eigs_num, which='LM', v0=start_vec)

    return values, vecs

