import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as anm
from scipy.interpolate import CubicSpline as cs
from scipy.interpolate import interp1d
from scipy import signal
from view import View
from PCA_interpolation import PCA_based_reconstruction as PCA

class data:
    def __init__(self, filename, time, elim_outlier=False, zeroisnan=False):
        self.filename = filename
        self.time = time
        self.__elim_outlier = elim_outlier
        self.__zeroisnan = zeroisnan
        self.timelist = []

        self.joint_name = []
        # x,y,z[joint][time]
        self.x = []
        self.y = []
        self.z = []

        self.__fig = None
        self.__ax = None
        self.__xrange = None
        self.__yrange = None
        self.__zrange = None

    

    def add_times(self, timelists):
        self.timelist = timelists

    def add_joints(self, joint_name, x, y, z, judgement=True):
        if not (len(x) == len(y) and len(x) == len(y) and len(y) == len(z)):
            print "this data({}) is not valid".format(joint_name)
            exit()


        X = np.array(x)
        Y = np.array(y)
        Z = np.array(z)
        if self.__zeroisnan:
            X[X == 0] = np.nan
            Y[Y == 0] = np.nan
            Z[Z == 0] = np.nan

        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if self.__elim_outlier:
            p75x, p25x = np.nanpercentile(X, [75, 25])
            IQRx = (p75x - p25x)

            p75y, p25y = np.nanpercentile(Y, [75, 25])
            IQRy = (p75y - p25y)

            p75z, p25z = np.nanpercentile(Z, [75, 25])
            IQRz = (p75z - p25z)

            bo = np.logical_or.reduce((X > p75x + IQRx * 1.5, X < p25x - IQRx * 1.5,
                                       Y > p75y + IQRy * 1.5, Y < p25y - IQRy * 1.5,
                                       Z > p75z + IQRz * 1.5, Z < p25z - IQRz * 1.5))
            X[bo] = np.nan
            Y[bo] = np.nan
            Z[bo] = np.nan

        if judgement and np.sum(np.isnan(X)) > self.time*0.25:
            return


        self.joint_name.append(joint_name)
        self.x.append(X)
        self.y.append(Y)
        self.z.append(Z)

        return

    def elim_outlier(self, padding=None):
        newx = []
        newy = []
        newz = []
        pad = padding
        if padding is None:
            pad = np.nan

        for x, y, z in zip(self.x, self.y, self.z):
            X = np.array(x)
            Y = np.array(y)
            Z = np.array(z)

            p75x, p25x = np.nanpercentile(X, [75, 25])
            IQRx = (p75x - p25x)

            p75y, p25y = np.nanpercentile(Y, [75, 25])
            IQRy = (p75y - p25y)

            p75z, p25z = np.nanpercentile(Z, [75, 25])
            IQRz = (p75z - p25z)

            bo = np.logical_or.reduce((X > p75x + IQRx * 3, X < p25x - IQRx * 3,
                                       Y > p75y + IQRy * 3, Y < p25y - IQRy * 3,
                                       Z > p75z + IQRz * 3, Z < p25z - IQRz * 3))

            X[bo] = pad
            Y[bo] = pad
            Z[bo] = pad
            """
            p75x, p25x = np.nanpercentile(X, [75, 25])
            IQRx = (p75x - p25x)

            p75y, p25y = np.nanpercentile(Y, [75, 25])
            IQRy = (p75y - p25y)

            p75z, p25z = np.nanpercentile(Z, [75, 25])
            IQRz = (p75z - p25z)

            bo = np.logical_or.reduce((X > p75x + IQRx * 1.5, X < p25x - IQRx * 1.5,
                                       Y > p75y + IQRy * 1.5, Y < p25y - IQRy * 1.5,
                                       Z > p75z + IQRz * 1.5, Z < p25z - IQRz * 1.5))

            X = np.array(x)
            Y = np.array(y)
            Z = np.array(z)

            X[bo] = pad
            Y[bo] = pad
            Z[bo] = pad
            """
            newx.append(X)
            newy.append(Y)
            newz.append(Z)

        self.x = newx
        self.y = newy
        self.z = newz

    def adjust_time(self, nan=None):

        initial_complete_time = []
        finish_complete_time = []

        for x in self.x:
            if nan is None:
                time = np.where(~np.isnan(x))[0]
            else:
                time = np.where(x == nan)[0]

            try:
                initial_complete_time.append(time[0])
                finish_complete_time.append(time[-1])
            except IndexError:
                raise IndexError("invalid joint")

        init_time = np.max(np.array(initial_complete_time))
        fin_time = np.min(np.array(finish_complete_time))

        self.time = fin_time - init_time
        self.timelist = self.timelist[init_time:fin_time]

        #transpose = lambda matrix: map(list, zip(*matrix))
        #self.x = transpose(transpose(self.x)[init_time:])
        self.x = list(np.array(self.x).T[init_time:fin_time].T)
        self.y = list(np.array(self.y).T[init_time:fin_time].T)
        self.z = list(np.array(self.z).T[init_time:fin_time].T)


    def filter(self, name='butter', fp=2, fs=20, gpass=1, gstop=40):
        fn = 120.0
        Wp = fp/fn
        Ws = fs/fn

        if name == 'butter':
            N, Wn = signal.buttord(Wp, Ws, gpass, gstop)
            b1, a1 = signal.butter(N, Wn, "low", analog=True)

        elif name == 'cheby1':
            N, Wn = signal.cheb1ord(Wp, Ws, gpass, gstop)
            b1, a1 = signal.cheby1(N, gpass, Wn, "low")

        elif name == 'cheby2':
            N, Wn = signal.cheb2ord(Wp, Ws, gpass, gstop)
            b1, a1 = signal.cheby2(N, gstop, Wn, "low")

        elif name == 'elip':
            N, Wn = signal.ellipord(Wp, Ws, gpass, gstop)
            b1, a1 = signal.ellip(N, gpass, gstop, Wn, "low")

        elif name == 'bessel':
            N = 4
            b1, a1 = signal.bessel(N, Ws, "low")

        newx = []
        newy = []
        newz = []
        for i in range(len(self.x)):
            newx.append(signal.filtfilt(b1, a1, self.x[i]))
            newy.append(signal.filtfilt(b1, a1, self.y[i]))
            newz.append(signal.filtfilt(b1, a1, self.z[i]))

            t = np.arange(self.x[0].shape[0])
            y = self.x[i]
            plt.subplot(1,2,1)
            plt.plot(t, y)

            plt.subplot(1, 2, 2)
            plt.plot(t, newx[i])

            plt.show()

        self.x = newx
        self.y = newy
        self.z = newz

    def interpolate(self, method='spline'):
        # interpolation indivisually
        if method == 'spline':
            newx = []
            newy = []
            newz = []
            for x, y, z in zip(self.x, self.y, self.z):
                time = np.where(~np.isnan(x))[0]

                spline_x = cs(time, x[time])
                spline_y = cs(time, y[time])
                spline_z = cs(time, z[time])

                time = [i for i in range(len(x))]

                newx.append(spline_x(time))
                newy.append(spline_y(time))
                newz.append(spline_z(time))
            self.x = newx
            self.y = newy
            self.z = newz
            return

        elif method == 'linear':
            newx = []
            newy = []
            newz = []
            for x, y, z in zip(self.x, self.y, self.z):
                time = np.where(~np.isnan(x))[0]

                interp1d_x = interp1d(time, x[time], fill_value='extrapolate')
                interp1d_y = interp1d(time, y[time], fill_value='extrapolate')
                interp1d_z = interp1d(time, z[time], fill_value='extrapolate')

                time = [i for i in range(len(x))]

                newx.append(interp1d_x(time))
                newy.append(interp1d_y(time))
                newz.append(interp1d_z(time))
            self.x = newx
            self.y = newy
            self.z = newz
            return

        elif method == 'PCA' or 'pca':
            self.x, self.y, self.z = PCA(self.x, self.y, self.z)
        else:
            print("warning: {0} is not defined as interpolation method".format(method))

    def transpose(self):
        return np.array(self.x).T, np.array(self.y).T, np.array(self.z).T

    def view(self, fps=240, savepath=None, saveonly=False, line_view=True, show_joint_name=False):
        view = View()
        if line_view:
            joint_name = self.joint_name
        else:
            joint_name = None
        view.show3d(self.x, self.y, self.z, fps, joint_name=joint_name, savepath=savepath, saveonly=saveonly, show_joint_name=show_joint_name)

    def xyz(self):#[joint][time][dim]
        xyz = np.array([np.array(self.x).T, np.array(self.y).T, np.array(self.z).T]).T
        return xyz