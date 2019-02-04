import platform
if platform.system() == 'Darwin':
    import matplotlib
    matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as anm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import numpy as np
import cv2
import sys
import os

class Visualization(object):
    def __init__(self):
        pass

    def show(self, x, y, xtime, ytime, title=None, savepath=None, legend=False, correspondLine=False, verbose=False):
        if type(x).__name__ != 'dict':
            raise ValueError("x must be \'dict\' instead of {0}".format(type(x).__name__))
        if type(y).__name__ != 'dict':
            raise ValueError("y must be \'dict\' instead of {0}".format(type(x).__name__))

        if len(x) != len(y):
            print("Warning: x's length{0} is not same to y's{1}. You may not be able to get right result...".format(len(x), len(y)))

        self.fig = plt.figure()
        plt.xlabel('reference')
        plt.ylabel('input')
        """
        if xtime > ytime:
            plt.xlim([0, xtime])
            plt.ylim([0, xtime])
        else:
            plt.xlim([0, ytime])
            plt.ylim([0, ytime])
        """
        plt.gca().set_aspect('equal', adjustable='box')
        plt.vlines([xtime], 0, ytime, linestyles='dashed')

        if not correspondLine:
            plt.plot([0, xtime], [0, xtime], 'black', linestyle='dashed')

        for joint in x.keys():
            plt.plot(x[joint], y[joint], label=joint)
            if correspondLine:
                plt.plot([0, xtime], [y[joint][0], xtime + y[joint][0]], 'black', linestyle='dashed')

        if legend:
            #plt.legend(ncol=2, bbox_to_anchor=(1.05, 1), prop={'size': 18})
            plt.legend(ncol=1)
        plt.rcParams["font.size"] = 18
        if title is not None:
            plt.title(title)

        if savepath is None:
            plt.show()
        else:
            if not os.path.isdir(os.path.dirname(savepath)):
                os.mkdir(os.path.dirname(savepath))
            #plt.savefig(savepath, bbox_inches="tight")
            plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.2)
            plt.savefig(savepath)
            if verbose:
                print('saved {0}'.format(savepath))

        return

    def show3d(self, x, y, z, jointNames, saveonly=False, title=None, savepath=None, fps=240, lines=None, verbose=False, colors=None): # x[time, joint] ,color must be normalized -1~1
        if saveonly:
            if savepath is None:
                raise ValueError("when you call save, you must set savepath")
            gui = gui3d(x, y, z, jointNames, fps, lines, colors)
            gui.setViewRange()
            gui.saveVideo(savepath)
            if verbose:
                print('saved {0}'.format(savepath))
            return
        app = QApplication(sys.argv)
        gui = gui3d(x, y, z, jointNames, fps, lines, colors)
        gui.setViewRange()
        gui.show()
        sys.exit(app.exec_())


class gui3d(QMainWindow):
    def __init__(self, x, y, z, joints, fps=240, lines=None, colors=None, parent=None):
        QMainWindow.__init__(self, parent)
        self.frame = 0
        self.x = x # [time][joint index]
        self.y = y
        self.z = z
        self.joints = joints
        self.lines = lines
        self.fps = fps
        self.colors = colors

        self.create_menu()
        self.create_mainframe()
        self.draw(fix=False)

    def draw(self, fix=False):
        if fix:
            azim = self.axes.azim
            elev = self.axes.elev
            xlim = list(self.axes.get_xlim())
            ylim = list(self.axes.get_ylim())
            zlim = list(self.axes.get_zlim())
            addlim = np.max([xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]])
            xlim[1] = xlim[0] + addlim
            ylim[1] = ylim[0] + addlim
            zlim[1] = zlim[0] + addlim
        # clear the axes and redraw the plot anew
        #
        self.axes.clear()
        plt.title('frame number=' + str(self.frame))
        self.axes.grid(self.grid_cb.isChecked())

        self.axes.set_xlabel('x')
        self.axes.set_ylabel('y')
        self.axes.set_zlabel('z')

        if fix:
            self.axes.set_xlim(xlim)
            self.axes.set_ylim(ylim)
            self.axes.set_zlim(zlim)
            self.axes.view_init(elev=elev, azim=azim)

        if self.colors is not None:
            self.scatter = [
                self.axes.scatter3D(self.x[self.frame, i], self.y[self.frame, i], self.z[self.frame, i], ".",
                                    color=self.colors[self.frame, i], picker=5) for i in range(len(self.joints))]
        else:
            self.scatter = [
                self.axes.scatter3D(self.x[self.frame, i], self.y[self.frame, i], self.z[self.frame, i], ".", edgecolor='black',
                                    color='black', picker=5) for i in range(len(self.joints))]
            pass

        if self.lines is not None:
            for line in self.lines:
                self.axes.plot([self.x[self.frame, line[0]], self.x[self.frame, line[1]]],
                               [self.y[self.frame, line[0]], self.y[self.frame, line[1]]],
                               [self.z[self.frame, line[0]], self.z[self.frame, line[1]]], "-", color='black')
        """
        if self.trajectory_line is not None:
            self.axes.lines.extend(self.trajectory_line)

        self.scatter = [
            self.axes.scatter3D(self.x[self.frame, i], self.y[self.frame, i], self.z[self.frame, i], ".",
                                color='blue', picker=5) for i in range(len(self.joints))]

        if self.now_select != -1:
            self.scatter[self.now_select] = self.axes.scatter3D(self.x[self.frame, self.now_select],
                                                                self.y[self.frame, self.now_select],
                                                                self.z[self.frame, self.now_select], ".",
                                                                color='red', picker=5)

        if self.leftdockwidget.check_showbone.isChecked():
            for bone1, bone2 in zip(self.bone1[self.frame], self.bone2[self.frame]):
                self.axes.plot([bone1[0], bone2[0]], [bone1[1], bone2[1]], [bone1[2], bone2[2]], "-",
                               color="black")
        """
        self.canvas.draw()

    def create_menu(self):
        self.filemenu = self.menuBar().addMenu("&File")

        save_action = QAction("Save video", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save)
        self.filemenu.addAction(save_action)

    def create_mainframe(self):
        self.main_frame = QWidget()

        # Create the mpl Figure and FigCanvas objects.
        # 5x4 inches, 100 dots-per-inch
        #
        self.dpi = 100
        self.fig = plt.figure()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)

        # Since we have only one plot, we can use add_axes
        # instead of add_subplot, but then the subplot
        # configuration tool in the navigation toolbar wouldn't
        # work.
        #
        self.axes = Axes3D(self.fig)

        # Bind the 'pick' event for clicking on one of the bars
        #
        #self.canvas.mpl_connect('pick_event', self.onclick)
        #self.canvas.setFocusPolicy(Qt.ClickFocus)
        #self.canvas.setFocus()
        #self.canvas.mpl_connect('key_press_event', self.onkey)
        #self.canvas.mpl_connect('key_release_event', self.onrelease)

        # Create the navigation toolbar, tied to the canvas
        #
        self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)

        # Other GUI controls
        #
        self.grid_cb = QCheckBox("Show &Grid")
        self.grid_cb.setChecked(True)
        self.grid_cb.stateChanged.connect(lambda: self.draw(fix=True))

        # x range selector
        self.groupxrange = QGroupBox("x")
        hboxX = QHBoxLayout()
        self.buttonXminus = QPushButton("<")
        self.buttonXminus.clicked.connect(lambda: self.rangeChanger("x", False))
        hboxX.addWidget(self.buttonXminus)

        self.buttonXplus = QPushButton(">")
        self.buttonXplus.clicked.connect(lambda: self.rangeChanger("x", True))
        hboxX.addWidget(self.buttonXplus)
        self.groupxrange.setLayout(hboxX)

        # y range selector
        self.groupyrange = QGroupBox("y")
        hboxY = QHBoxLayout()
        self.buttonYminus = QPushButton("<")
        self.buttonYminus.clicked.connect(lambda: self.rangeChanger("y", False))
        hboxY.addWidget(self.buttonYminus)

        self.buttonYplus = QPushButton(">")
        self.buttonYplus.clicked.connect(lambda: self.rangeChanger("y", True))
        hboxY.addWidget(self.buttonYplus)
        self.groupyrange.setLayout(hboxY)

        # z range changer
        self.groupzrange = QGroupBox("z")
        hboxZ = QHBoxLayout()
        self.buttonZminus = QPushButton("<")
        self.buttonZminus.clicked.connect(lambda: self.rangeChanger("z", False))
        hboxZ.addWidget(self.buttonZminus)

        self.buttonZplus = QPushButton(">")
        self.buttonZplus.clicked.connect(lambda: self.rangeChanger("z", True))
        hboxZ.addWidget(self.buttonZplus)
        self.groupzrange.setLayout(hboxZ)

        #self.groupxrange.setEnabled(False)
        #self.groupyrange.setEnabled(False)
        #self.groupzrange.setEnabled(False)
        #
        # Layout with box sizers
        #
        hbox = QHBoxLayout()

        for w in [self.grid_cb, self.groupxrange, self.groupyrange, self.groupzrange]:
            hbox.addWidget(w)
            hbox.setAlignment(w, Qt.AlignVCenter)

        #
        # slider
        #
        self.slider = QSlider(Qt.Horizontal)
        #self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.sliderValueChanged)
        self.slider.setMaximum(self.x.shape[0] - 1)
        self.slider.setMinimum(0)

        vbox = QVBoxLayout()
        vbox.addWidget(self.slider)
        vbox.addWidget(self.canvas)
        vbox.addWidget(self.mpl_toolbar)
        vbox.addLayout(hbox)

        self.main_frame.setLayout(vbox)
        self.setCentralWidget(self.main_frame)

    def rangeChanger(self, coordinates, plus):
        ticks = eval("self.axes.get_{0}ticks()".format(coordinates))

        if plus:
            width = ticks[1] - ticks[0]
        else:
            width = ticks[0] - ticks[1]

        lim = eval("self.axes.get_{0}lim()".format(coordinates))
        eval("self.axes.set_{0}lim(lim + width)".format(coordinates))

        self.draw(fix=True)

    def sliderValueChanged(self):
        self.frame = self.slider.value()
        self.draw(fix=True)

    def save(self):
        filters = "MP4 files(*.MP4)"
        # selected_filter = "CSV files(*.csv)"
        savepath, extension = QFileDialog.getSaveFileNameAndFilter(self, 'Save file', '', filters)

        savepath = str(savepath).encode()
        extension = str(extension).encode()
        # print(extension)
        if savepath != "":
            savepath += '.MP4'

            self.saveVideo(savepath)

            QMessageBox.information(self, "Saved", "Saved to {0}".format(savepath))

    def setViewRange(self):
        azim = self.axes.azim
        elev = self.axes.elev
        xlim = list((np.nanmin(self.x), np.nanmax(self.x)))
        ylim = list((np.nanmin(self.y), np.nanmax(self.y)))
        zlim = list((np.nanmin(self.z), np.nanmax(self.z)))
        addlim = np.max([xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]])
        xlim[1] = xlim[0] + addlim
        ylim[1] = ylim[0] + addlim
        zlim[1] = zlim[0] + addlim

        self.axes.set_xlim(xlim)
        self.axes.set_ylim(ylim)
        self.axes.set_zlim(zlim)
        self.axes.view_init(elev=elev, azim=azim)

    def saveVideo(self, savepath):
        size = (600, 400)
        sys.stdout.write('\r')
        sys.stdout.flush()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(savepath, fourcc, self.fps, size)

        azim = self.axes.azim
        elev = self.axes.elev
        xlim = list(self.axes.get_xlim())
        ylim = list(self.axes.get_ylim())
        zlim = list(self.axes.get_zlim())
        addlim = np.max([xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]])
        xlim[1] = xlim[0] + addlim
        ylim[1] = ylim[0] + addlim
        zlim[1] = zlim[0] + addlim

        self.axes.set_xlim(xlim)
        self.axes.set_ylim(ylim)
        self.axes.set_zlim(zlim)
        self.axes.view_init(elev=elev, azim=azim)
        sys.stdout.write('\n')
        sys.stdout.flush()
        for frame in range(self.x.shape[0]):
            percent = int((frame + 1.0) * 100 / self.x.shape[0])
            sys.stdout.write(
                '\r|{0}| {1}% finished'.format('#' * int(percent * 0.2) + '-' * (20 - int(percent * 0.2)), percent))
            sys.stdout.flush()
            self.__update3d(frame, xlim, ylim, zlim)

            # convert canvas to image
            self.axes.figure.canvas.draw()
            """
            img = np.fromstring(self.axes.figure.canvas.tostring_rgb(), dtype=np.uint8,
                                sep='')

            print (self.axes.figure.canvas.get_width_height()[::-1])
            print (self.axes.figure.canvas.get_width_height())
            exit()
            img = img.reshape(self.axes.figure.canvas.get_width_height()[::-1] + (3,))
            """
            width, height = self.fig.get_size_inches() * self.fig.get_dpi()
            img = np.fromstring(self.fig.canvas.tostring_rgb(), dtype='uint8', sep='').reshape(int(height), int(width), 3)
            # img is rgb, convert to opencv's default bgr
            img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), size)

            video.write(img)

            # display image with opencv or any operation you like
            # cv2.imshow("plot", img)
            # k = cv2.waitKey(int(100*1.0/fps))
            # if k == ord('q'):
            #    show = False
            #    break

        video.release()
        sys.stdout.write('\rsaved to {0}\n'.format(savepath))
        sys.stdout.flush()


    def __update3d(self, frame, xrange, yrange, zrange):
        if frame != 0:
            self.axes.cla()

            self.axes.set_xlim(xrange)
            self.axes.set_ylim(yrange)
            self.axes.set_zlim(zrange)

        if self.colors is not None:
            self.scatter = [
                self.axes.scatter3D(self.x[frame, i], self.y[frame, i], self.z[frame, i], ".",
                                    color=self.colors[frame, i], picker=5) for i in range(len(self.joints))]
        else:
            self.scatter = [
                self.axes.scatter3D(self.x[frame, i], self.y[frame, i], self.z[frame, i], ".", edgecolor='black',
                                    color='black', picker=5) for i in range(len(self.joints))]
        #self.axes.scatter3D(self.x[frame], self.y[frame], self.z[frame], ".")

        if self.lines is not None:
            for line in self.lines:
                self.axes.plot([self.x[frame, line[0]], self.x[frame, line[1]]],
                               [self.y[frame, line[0]], self.y[frame, line[1]]],
                               [self.z[frame, line[0]], self.z[frame, line[1]]], "-", color='black')

        plt.title('frame:{0}'.format(frame))

