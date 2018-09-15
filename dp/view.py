import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as anm
import numpy as np
import cv2
import sys
import os

class Visualization(object):
    def __init__(self):
        pass

    def show(self, x, y, xtime, ytime, title=None, savepath=None):
        self.fig = plt.figure()
        plt.xlabel('reference')
        plt.ylabel('input')
        if xtime > ytime:
            plt.xlim([0, xtime])
            plt.ylim([0, xtime])
        else:
            plt.xlim([0, ytime])
            plt.ylim([0, ytime])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.vlines([xtime], 0, ytime, linestyles='dashed')
        tmp_x = np.linspace(0, xtime, xtime + 1)
        # plt.plot([0, self.input.time], [0, self.input.time], 'black', linestyle='dashed')
        plt.plot([0, xtime], [y[0], xtime + y[0]], 'black', linestyle='dashed')
        plt.plot(x, y, label='Matching Path')

        plt.legend()

        if title is not None:
            plt.title(title)

        if savepath is None:
            plt.show()
        else:
            if not os.path.isdir(os.path.dirname(savepath)):
                os.mkdir(os.path.dirname(savepath))
            plt.savefig(savepath)

        return



class View:
    def __init__(self):
        self.__fig = None
        self.__ax = None

    def show3d(self, X, Y, Z, fps=30, joint_name=None, show_joint_name=False, savepath=None, saveonly=False): # X,Y,Z[time][joint]
        x_ = np.array(X)
        y_ = np.array(Y)
        z_ = np.array(Z)

        if not (x_.shape == y_.shape == z_.shape):
            print("array size error")
            print("x:{0}".format(x_.shape))
            print("y:{0}".format(y_.shape))
            print("z:{0}".format(z_.shape))

            exit()

        self.__fig = plt.figure()
        self.__ax = Axes3D(self.__fig)
        self.__ax.set_xlabel('x')
        self.__ax.set_ylabel('y')
        self.__ax.set_zlabel('z')

        line_lower = [['LTOE', 'LANK'], ['LTIB', 'LANK'], ['LASI', 'LPSI'],  # around ankle
                      ['RTOE', 'RANK'], ['RTIB', 'RANK'], ['RASI', 'RPSI'],  # "
                      ['LASI', 'RASI'], ['LPSI', 'RPSI'], ['LHEE', 'LANK'], ['RHEE', 'RANK'], ['LHEE', 'LTOE'],
                      ['RHEE', 'RTOE'],  # around hip
                      ['LHEE', 'LTIB'], ['RHEE', 'RTIB'],  # connect ankle to knee
                      ['LKNE', 'LTIB'], ['LKNE', 'LTHI'], ['LASI', 'LTHI'], ['LPSI', 'LTHI'],  # connect knee to hip
                      ['RKNE', 'RTIB'], ['RKNE', 'RTHI'], ['RASI', 'RTHI'], ['RPSI', 'RTHI'],  # "
                      ['LPSI', 'T10'], ['RPSI', 'T10'], ['LASI', 'STRN'], ['RASI', 'STRN'],  # conncet lower and upper
                      ]
        line_upper = [['LFHD', 'LBHD'], ['RFHD', 'RBHD'], ['LFHD', 'RFHD'], ['LBHD', 'RBHD'],  # around head
                      ['LBHD', 'C7'], ['RBHD', 'C7'], ['C7', 'CLAV'], ['CLAV', 'LSHO'], ['CLAV', 'RSHO'],  # connect head to shoulder
                      ['LSHO', 'LBAK'], ['RSHO', 'RBAK'], ['RBAK', 'LBAK'],  # around shoulder
                      ['LWRA', 'LFIN'], ['LWRA', 'LFIN'], ['LWRA', 'LWRB'], ['LWRA', 'LFRM'], ['LWRB', 'LFRM'],  # around wrist
                      ['RWRA', 'RFIN'], ['RWRA', 'RFIN'], ['RWRA', 'RWRB'], ['RWRA', 'RFRM'], ['RWRB', 'RFRM'],  # "
                      ['LELB', 'LRFM'], ['LELB', 'LUPA'], ['LELB', 'LFIN'], ['LUPA', 'LSHO'],  # connect elbow to wrist, connect elbow to shoulder
                      ['RELB', 'RRFM'], ['RELB', 'RUPA'], ['RELB', 'RFIN'], ['RUPA', 'RSHO'],  # "
                      ['LSHO', 'STRN'], ['RSHO', 'STRN'], ['LBAK', 'T10'], ['RBAK', 'T10'],  # connect shoulder to torso
                      ]

        # show mean movement
        x = x_.flatten()
        y = y_.flatten()
        z = z_.flatten()

        xrange = [np.nanmin(x), np.nanmax(x)]
        yrange = [np.nanmin(y), np.nanmax(y)]
        zrange = [np.nanmin(z), np.nanmax(z)]

        aspectFT = np.array([xrange[1] - xrange[0], yrange[1] - yrange[0],
                             zrange[1] - zrange[0]])
        max = aspectFT.max()
        xrange[1] += max
        yrange[1] += max
        zrange[1] += max

        # ax.set_xlim(xrange)
        # ax.set_ylim(yrange)
        # ax.set_zlim(zrange)

        x = x_.T
        y = y_.T
        z = z_.T

        # string(joint name) to number(index)
        line_index_lower = None
        line_index_upper = None

        if joint_name is not None:
            line_index_lower = []
            line_index_upper = []
            # extract initial part of joint name 'skelton 04:'hip
            init_string = joint_name[0]
            init_string = init_string[:init_string.index(':') + 1]
            for line in line_lower:
                try:
                    line_index_lower.append(
                        [joint_name.index(init_string + line[0]), joint_name.index(init_string + line[1])])
                except:
                    continue
            for line in line_upper:
                try:
                    line_index_upper.append(
                        [joint_name.index(init_string + line[0]), joint_name.index(init_string + line[1])])
                except:
                    continue

        show = True
        size = (600, 400)
        if savepath is not None:
            fourcc = cv2.VideoWriter_fourcc(*'MPEG')
            video = cv2.VideoWriter(savepath, fourcc, fps, size)
            for frame in range(len(x)):
                percent = int((frame + 1.0)*100/len(x))
                sys.stdout.write('\r|{0}| {1}% finished'.format('#'*int(percent*0.2)+'-'*(20 - int(percent*0.2)), percent))
                sys.stdout.flush()
                self.__update3d(frame, x, y, z, xrange, yrange, zrange,
                                joint_name, line_index_lower, line_index_upper, show_joint_name)

                # convert canvas to image
                self.__ax.figure.canvas.draw()
                img = np.fromstring(self.__ax.figure.canvas.tostring_rgb(), dtype=np.uint8,
                                    sep='')
                img = img.reshape(self.__ax.figure.canvas.get_width_height()[::-1] + (3,))

                # img is rgb, convert to opencv's default bgr
                img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), size)

                video.write(img)
                """
                # display image with opencv or any operation you like
                cv2.imshow("plot", img)
                k = cv2.waitKey(int(100*1.0/fps))
                if k == ord('q'):
                    show = False
                    break
                """
            video.release()
            print("\nsaved to {0}".format(savepath))

            if saveonly:
                return

        ani = anm.FuncAnimation(self.__fig, self.__update3d, fargs=(x, y, z, xrange, yrange, zrange,
                                                         joint_name, line_index_lower, line_index_upper, show_joint_name),
                                interval=1.0/fps, frames=len(x))  # interval ms
        """
        if savepath is not None:
            print("saving now...")
            ani.save(savepath)
            print("saved to {0}".format(savepath))
            if saveonly:
                return
        """
        plt.show()

    def __update3d(self, frame, x, y, z, xrange, yrange, zrange, joint_name=None, line_lower=None, line_upper= None, show_joint_name=False):
        if frame != 0:
            self.__ax.cla()

        if not show_joint_name:
            self.__ax.set_xlim(xrange)
            self.__ax.set_ylim(yrange)
            self.__ax.set_zlim(zrange)

        self.__ax.scatter3D(x[frame], y[frame], z[frame], ".")

        if line_lower is not None:
            for line in line_lower:
                self.__ax.plot([x[frame, line[0]], x[frame, line[1]]],
                        [y[frame, line[0]], y[frame, line[1]]],
                        [z[frame, line[0]], z[frame, line[1]]], "-", color='black')
        if line_upper is not None:
            for line in line_upper:
                self.__ax.plot([x[frame, line[0]], x[frame, line[1]]],
                        [y[frame, line[0]], y[frame, line[1]]],
                        [z[frame, line[0]], z[frame, line[1]]], "-", color='black')

        if show_joint_name:
            for joint_index, joint in enumerate(joint_name):
                self.__ax.scatter3D(x[frame][joint_index], y[frame][joint_index], z[frame][joint_index], ".")
                self.__ax.text(x[frame][joint_index], y[frame][joint_index], z[frame][joint_index], str(joint[12:]))
        plt.title('frame:{0}'.format(frame))

    def show2d(self):
        self.__fig = plt.figure()