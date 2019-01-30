from dp.dp import DP
from dp.data import Data
from dp.utils import csvReader, contexts
from dp.contextdp import SyncContextDP, AsyncContextDP
import sys
import os
import platform
if platform.system() == 'Darwin':
    import matplotlib
    matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import numpy as np
import cv2


class DPgui(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)

        self.setConfig()
        self.initUI()

    def setConfig(self):
        # initialixzation
        self.caches = {'prevDirVolleyball':None, 'prevDirBaseball':None,
                        'prevContextsDirVolleyball':None, 'prevContextsDirBaseball':None}
        self.configs = {'constraint':None, 'baseBallFps-int':None}

        self.done = False
        self.colors = None
        self.frame = 0
        self.x = None
        self.y = None
        self.z = None
        self.joints = None
        self.lines = None
        self.frame_max = 0

        if not os.path.exists('./.config'):
            os.mkdir('.config')
        # cache
        if not os.path.exists('./.config/gui.cache'):
            with open('./.config/gui.cache', 'w') as f:
                for cacheName, cacheValue in self.caches.items():
                    f.write('{0},{1},\n'.format(cacheName, cacheValue))

        with open('./.config/gui.cache', 'r') as f:
            lines = f.readlines()
            for line in lines:
                tmp = line.split(',')
                if tmp[1] == 'None':
                    exec ('self.caches[\'{0}\'] = None'.format(tmp[0]))
                else:
                    exec ('self.caches[\'{0}\'] = \'{1}\''.format(tmp[0], tmp[1]))

        # config
        if not os.path.exists('./.config/gui.config'):
            self.writeConfig()
        self.readConfig()

    def initUI(self):
        self.create_main_frame()
        self.create_menu()
        self.setleftDock()

    def create_menu(self):
        # file
        self.file_menu = self.menuBar().addMenu("&File")

        #input_action = self.create_action("&Input", slot=self.input_trcfile, shortcut="Ctrl+I", tip="Input csv file")
        #self.add_actions(self.file_menu, (input_action,))

        saveVideo_action = self.create_action("Save as video", slot=self.saveVideo, shortcut="Ctrl+S", tip="save DP result as video")
        self.add_actions(self.file_menu, (saveVideo_action,))
        saveVideo_action.setEnabled(False)

        quit_action = self.create_action("&Quit", slot=self.close, shortcut="Ctrl+Q", tip="Close the application")
        self.add_actions(self.file_menu, (None, quit_action))

        # help
        self.help_menu = self.menuBar().addMenu("&Help")
        about_action = self.create_action("&About", shortcut='F1', slot=self.show_about, tip='About the demo')
        self.add_actions(self.help_menu, (about_action,))

        # setting
        self.setting_menu = self.menuBar().addMenu("&Setting")
        preference_action = self.create_action("Preferences", shortcut='Ctrl+,', slot=self.preference, tip='set preferences')
        self.add_actions(self.setting_menu, (preference_action,))

        # edit
        self.edit_menu = self.menuBar().addMenu("&Edit")

        nextframe_action = self.create_action("Next Frame", slot=self.nextframe, shortcut="Ctrl+N", tip="show next frame")
        self.add_actions(self.edit_menu, (nextframe_action,))

        previousframe_action = self.create_action("Previous Frame", slot=self.previousframe, shortcut="Ctrl+P", tip="show previous frame")
        self.add_actions(self.edit_menu, (previousframe_action,))

    def create_main_frame(self):
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
        self.canvas.setFocusPolicy(Qt.ClickFocus)
        self.canvas.setFocus()
        self.canvas.mpl_connect('key_press_event', self.onkey)
        self.canvas.mpl_connect('key_release_event', self.onrelease)
        # Create the navigation toolbar, tied to the canvas
        #
        self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)

        # Other GUI controls
        #
        self.grid_cb = QCheckBox("Show &Grid")
        self.grid_cb.setChecked(True)
        self.grid_cb.stateChanged.connect(self.draw)

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

        self.groupxrange.setEnabled(False)
        self.groupyrange.setEnabled(False)
        self.groupzrange.setEnabled(False)
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
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.sliderValueChanged)
        self.slider.setMaximum(0)
        self.slider.setMinimum(0)

        vbox = QVBoxLayout()
        vbox.addWidget(self.slider)
        vbox.addWidget(self.canvas)
        vbox.addWidget(self.mpl_toolbar)
        vbox.addLayout(hbox)

        self.main_frame.setLayout(vbox)
        self.setCentralWidget(self.main_frame)

    # menu event
    def add_actions(self, target, actions):
        for action in actions:
            if action is None:
                target.addSeparator()
            else:
                target.addAction(action)

    def create_action(self, text, slot=None, shortcut=None,
                      icon=None, tip=None, checkable=False, ):
        action = QAction(text, self)
        if icon is not None:
            action.setIcon(QIcon(":/%s.png" % icon))
        if shortcut is not None:
            action.setShortcut(shortcut)
        if tip is not None:
            action.setToolTip(tip)
            action.setStatusTip(tip)
        if slot is not None:
            action.triggered.connect(slot)
        if checkable:
            action.setCheckable(True)
        return action

    def show_about(self):  # show detail of this application
        msg = """ A demo of using PyQt with matplotlib:

         * Use the matplotlib navigation bar
         * Add values to the text box and press Enter (or click "Draw")
         * Show or hide the grid
         * Drag the slider to modify the width of the bars
         * Save the plot to a file using the File menu
         * Click on a bar to receive an informative message
        """
        QMessageBox.about(self, "About the demo", msg.strip())

    def saveVideo(self):
        pass

    def preference(self):
        preferenceDialog = PreferenceDialog(self)
        preferenceDialog.setWindowModality(Qt.ApplicationModal)
        preferenceDialog.show()

    # config
    def readConfig(self):
        with open('./.config/gui.config', 'r') as f:
            lines = f.readlines()
            for line in lines:
                tmp = line.split(',')
                if tmp[1] == 'None':
                    exec ('self.configs[\'{0}\'] = None'.format(tmp[0]))
                else:
                    exec ('self.configs[\'{0}\'] = \'{1}\''.format(tmp[0], tmp[1]))

    def writeConfig(self):
        with open('./.config/gui.config', 'w') as f:
            for configName, configValue in self.configs.items():
                f.write('{0},{1},\n'.format(configName, configValue))

    # data
    def reset(self):
        self.done = False
        self.colors = None

        self.x = None
        self.y = None
        self.z = None
        self.joints = None
        self.lines = None
        self.frame_max = 0

        self.axes.clear()

        self.slider.setEnabled(False)
        self.slider.setMaximum(0)
        self.groupxrange.setEnabled(False)
        self.groupyrange.setEnabled(False)
        self.groupzrange.setEnabled(False)
        self.leftdockwidget.buttonPlay.setEnabled(False)
        self.leftdockwidget.buttonPause.setEnabled(False)

    # left dock
    def setleftDock(self):
        self.leftdock = QDockWidget(self)
        self.leftdock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.leftdock.setFloating(False)

        self.leftdockwidget = LeftDockWidget(self)
        self.leftdock.setWidget(self.leftdockwidget)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.leftdock)

    # draw
    def draw(self):
        if self.done:
            # save previous view point
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

            # restore previous view point
            self.axes.set_xlim(xlim)
            self.axes.set_ylim(ylim)
            self.axes.set_zlim(zlim)
            self.axes.view_init(elev=elev, azim=azim)

            self.scatter = [
                self.axes.scatter3D(self.x[self.frame, i], self.y[self.frame, i], self.z[self.frame, i], ".",
                                    color=self.colors[self.frame, i]) for i in range(len(self.joints))]
            for line in self.lines:
                self.axes.plot([self.x[self.frame, line[0]], self.x[self.frame, line[1]]],
                                [self.y[self.frame, line[0]], self.y[self.frame, line[1]]],
                                [self.z[self.frame, line[0]], self.z[self.frame, line[1]]], "-", color='black')
            self.canvas.draw()

    def initialDraw(self, inpData, colors):
        if type(colors).__name__ == 'list':
            e = colors[0]
            tb = colors[1]
            QMessageBox.critical(self, "Caution", "Unexpected error occured:{0}, maybe invalid combination".format(e.with_traceback(tb)))
            return
        self.done = True
        self.colors = colors

        data = np.array(list(inpData.joints.values()))  # [joint index][time][dim]
        self.x = data[:, :, 0].T
        self.y = data[:, :, 1].T
        self.z = data[:, :, 2].T
        self.joints = inpData.joints
        self.lines = inpData.lines
        self.frame_max = self.x.shape[0]
        # clear the axes and redraw the plot anew
        #
        self.axes.clear()
        plt.title('frame number=' + str(self.frame))
        self.axes.grid(self.grid_cb.isChecked())
        self.setViewRange()

        self.axes.set_xlabel('x')
        self.axes.set_ylabel('y')
        self.axes.set_zlabel('z')

        self.scatter = [
            self.axes.scatter3D(self.x[self.frame, i], self.y[self.frame, i], self.z[self.frame, i], ".",
                                color=self.colors[self.frame, i]) for i in range(len(self.joints))]
        for line in self.lines:
            self.axes.plot([self.x[self.frame, line[0]], self.x[self.frame, line[1]]],
                           [self.y[self.frame, line[0]], self.y[self.frame, line[1]]],
                           [self.z[self.frame, line[0]], self.z[self.frame, line[1]]], "-", color='black')
        self.canvas.draw()

        self.slider.setValue(0)
        self.draw()

        self.slider.setEnabled(True)
        self.slider.setMaximum(self.frame_max - 1)
        self.groupxrange.setEnabled(True)
        self.groupyrange.setEnabled(True)
        self.groupzrange.setEnabled(True)
        self.leftdockwidget.buttonPlay.setEnabled(True)

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

    # event
    def onrelease(self, event):
        if self.done:
            if self.xbutton and event.key == '.':
                self.rangeChanger("x", True)
            elif self.xbutton and event.key == ',':
                self.rangeChanger("x", False)
            elif self.ybutton and event.key == '.':
                self.rangeChanger("y", True)
            elif self.ybutton and event.key == ',':
                self.rangeChanger("y", False)
            elif self.zbutton and event.key == '.':
                self.rangeChanger("z", True)
            elif self.zbutton and event.key == ',':
                self.rangeChanger("z", False)

            if event.key == 'x':
                self.xbutton = False
            elif event.key == 'y':
                self.ybutton = False
            elif event.key == 'z':
                self.zbutton = False

    def onkey(self, event):
        if self.done:
            if event.key == ',' and self.frame != 0 and not self.xbutton and not self.ybutton and not self.zbutton:
                self.frame += -1
                self.sliderSetValue(self.frame)
            elif event.key == '.' and self.frame != self.frame_max and not self.xbutton and not self.ybutton and not self.zbutton:
                self.frame += 1
                self.sliderSetValue(self.frame)
            elif event.key == 'q':
                result = QMessageBox.warning(self, "Will you quit?",
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if result == QMessageBox.Yes:
                    plt.close(event.canvas.figure)
                else:
                    pass
            elif event.key == 'x':
                self.xbutton = True
            elif event.key == 'y':
                self.ybutton = True
            elif event.key == 'z':
                self.zbutton = True

            self.draw()

    # frame
    def nextframe(self):
        if self.done:
            if self.frame != self.frame_max:
                self.frame += 1
                self.sliderSetValue(self.frame)

    def previousframe(self):
        if self.done:
            if self.frame != self.frame_max:
                self.frame += -1
                self.sliderSetValue(self.frame)

    def rangeChanger(self, coordinates, plus):
        ticks = eval("self.axes.get_{0}ticks()".format(coordinates))

        if plus:
            width = ticks[1] - ticks[0]
        else:
            width = ticks[0] - ticks[1]

        lim = list(eval("self.axes.get_{0}lim()".format(coordinates)))
        lim += width
        exec ("self.axes.set_{0}lim(lim)".format(coordinates))

        self.draw()

    def sliderValueChanged(self):
        self.frame = self.slider.value()
        self.draw()

    def sliderSetValue(self, value):
        self.slider.setValue(value)

    def closeEvent(self, a0: QCloseEvent):
        # cache
        with open('./.config/gui.cache', 'w') as f:
            for cacheName, cacheValue in self.caches.items():
                f.write('{0},{1},\n'.format(cacheName, cacheValue))

        # config
        self.writeConfig()


class LeftDockWidget(QWidget):
    def __init__(self, parent):
        QWidget.__init__(self, parent)
        self.parent = parent

        self.calcType = 'Independent'
        self.contextsSet = {'type': 'Default', 'contexts': contexts('Baseball')}
        self.refPath = None
        self.inpPath = None

        self.initUI()


    def initUI(self):
        vbox = QVBoxLayout()

        self.groupConfig = QGroupBox("DP")
        vboxConfig = QVBoxLayout()

        self.labelCalculationType = QLabel()
        self.labelCalculationType.setText("Calculation Type")
        vboxConfig.addWidget(self.labelCalculationType)

        self.comboBoxCalculationType = QComboBox()
        self.comboBoxCalculationType.addItems(["Independent", "Synchronous Contexts", "Asynchronous Contexts"])
        self.comboBoxCalculationType.currentIndexChanged.connect(self.comboBoxCalculationTypeChanged)
        vboxConfig.addWidget(self.comboBoxCalculationType)

        self.labelSkeltonType = QLabel()
        self.labelSkeltonType.setText("Skelton Type")
        vboxConfig.addWidget(self.labelSkeltonType)

        self.comboBoxSkeltonType = QComboBox()
        self.comboBoxSkeltonType.addItems(["Baseball", "Volleyball"])
        self.comboBoxSkeltonType.currentIndexChanged.connect(self.comboBoxSkeltonTypeChanged)
        vboxConfig.addWidget(self.comboBoxSkeltonType)

        self.labelContextsType = QLabel()
        self.labelContextsType.setText("Default")
        self.labelContextsType.setEnabled(False)
        vboxConfig.addWidget(self.labelContextsType)

        self.buttonManuallyReadContexts = QPushButton("Manually Read Contexts")
        self.buttonManuallyReadContexts.clicked.connect(self.manuallyReadContextsClicked)
        self.buttonManuallyReadContexts.setEnabled(False)
        vboxConfig.addWidget(self.buttonManuallyReadContexts)

        self.labelReferencePatternPath = QLabel()
        self.labelReferencePatternPath.setText("Reference Pattern Path")
        vboxConfig.addWidget(self.labelReferencePatternPath)

        self.labelRealRefPath = QLabel()
        self.labelRealRefPath.setText("No selected")
        vboxConfig.addWidget(self.labelRealRefPath)

        self.buttonOpenRefPath = QPushButton("Open")
        self.buttonOpenRefPath.clicked.connect(lambda: self.openClicked(True))
        vboxConfig.addWidget(self.buttonOpenRefPath)

        self.labelInputPatternPath = QLabel()
        self.labelInputPatternPath.setText("Input Pattern Path")
        vboxConfig.addWidget(self.labelInputPatternPath)

        self.labelRealInpPath = QLabel()
        self.labelRealInpPath.setText("No selected")
        vboxConfig.addWidget(self.labelRealInpPath)

        self.buttonOpenInpPath = QPushButton("Open")
        self.buttonOpenInpPath.clicked.connect(lambda: self.openClicked(False))
        vboxConfig.addWidget(self.buttonOpenInpPath)

        self.buttonDone = QPushButton("Done!")
        self.buttonDone.clicked.connect(self.doneClicked)
        self.buttonDone.setEnabled(False)
        vboxConfig.addWidget(self.buttonDone)

        # add group into vbox
        self.groupConfig.setLayout(vboxConfig)
        vbox.addWidget(self.groupConfig)

        self.groupVisualization = QGroupBox("Visualization")
        vboxVisualization = QVBoxLayout()

        self.labelFps = QLabel()
        self.labelFps.setText("FPS")
        vboxVisualization.addWidget(self.labelFps)

        qintValidator = QIntValidator()
        self.lineeditFps = QLineEdit()
        self.lineeditFps.setValidator(qintValidator)
        self.lineeditFps.setText("240")
        vboxVisualization.addWidget(self.lineeditFps)

        self.labelMaximumGapTime = QLabel()
        self.labelMaximumGapTime.setText("Maximum Gap Time(s)")
        vboxVisualization.addWidget(self.labelMaximumGapTime)

        qdoubleValidator = QDoubleValidator()
        self.lineEditMaxGapTime = QLineEdit()
        self.lineEditMaxGapTime.setValidator(qdoubleValidator)
        self.lineEditMaxGapTime.setText("0.1")
        vboxVisualization.addWidget(self.lineEditMaxGapTime)

        self.hboxPlayPause = QHBoxLayout()
        self.buttonPlay = QPushButton("Play")
        self.buttonPlay.clicked.connect(self.play)
        self.buttonPlay.setEnabled(False)
        self.hboxPlayPause.addWidget(self.buttonPlay)

        self.buttonPause = QPushButton("Pause")
        self.buttonPause.clicked.connect(self.pause)
        self.buttonPause.setEnabled(False)
        self.hboxPlayPause.addWidget(self.buttonPause)

        vboxVisualization.addLayout(self.hboxPlayPause)

        # add group into vbox
        self.groupVisualization.setLayout(vboxVisualization)
        vbox.addWidget(self.groupVisualization)

        self.setLayout(vbox)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateVideo)

    # event
    def comboBoxCalculationTypeChanged(self):
        if 'Contexts' in str(self.comboBoxCalculationType.currentText()):
            self.labelContextsType.setEnabled(True)
            self.buttonManuallyReadContexts.setEnabled(True)
        else:
            self.labelContextsType.setEnabled(False)
            self.buttonManuallyReadContexts.setEnabled(False)

    def comboBoxSkeltonTypeChanged(self):
        self.refPath = None
        self.inpPath = None
        self.labelRealRefPath.setText("No selected")
        self.labelRealInpPath.setText("No selected")

        self.contextsSet = {'type': 'Default', 'contexts': contexts(str(self.comboBoxSkeltonType.currentText()))}
        self.labelContextsType.setText(self.contextsSet['type'])

    def openClicked(self, ref):
        basedir = ''
        if self.comboBoxSkeltonType.currentText() == 'Volleyball':
            filters = "TRC files(*.trc)"
            if self.parent.caches['prevDirVolleyball'] is not None:
                basedir = self.parent.caches['prevDirVolleyball']
        elif self.comboBoxSkeltonType.currentText() == 'Baseball':
            filters = "CSV files(*.csv)"
            if self.parent.caches['prevDirBaseball'] is not None:
                basedir = self.parent.caches['prevDirBaseball']
        else:
            filters = "TRC files(*.trc)"

        if ref:
            refPath, __ = QFileDialog.getOpenFileName(self, 'load file', basedir, filters)
            if refPath != "":
                self.refPath = refPath
                self.labelRealRefPath.setText(os.path.basename(self.refPath))
                exec('self.parent.caches[\'prevDir{0}\'] = os.path.dirname(self.refPath)'.format(self.comboBoxSkeltonType.currentText()))
            else:
                self.refPath = None
                self.labelRealRefPath.setText("No selected")


        else:
            inpPath, __ = QFileDialog.getOpenFileName(self, 'load file', basedir, filters)
            if inpPath != "":
                self.inpPath = inpPath
                self.labelRealInpPath.setText(os.path.basename(self.inpPath))
                exec('self.parent.caches[\'prevDir{0}\'] = os.path.dirname(self.inpPath)'.format(self.comboBoxSkeltonType.currentText()))
            else:
                self.inpPath = None
                self.labelRealInpPath.setText("No selected")

        if self.refPath is not None and self.inpPath is not None:
            self.buttonDone.setEnabled(True)
        else:
            self.buttonDone.setEnabled(False)

    def doneClicked(self):
        if self.comboBoxSkeltonType.currentText() == 'Baseball':
            refData = csvReader(os.path.basename(self.refPath), os.path.dirname(self.refPath))
            if refData is None:
                QMessageBox.critical(self, "Caution", "{0} cannot be loaded".format(self.refPath))
                return
            inpData = csvReader(os.path.basename(self.inpPath), os.path.dirname(self.inpPath))
            if inpData is None:
                QMessageBox.critical(self, "Caution", "{0} cannot be loaded".format(self.inpPath))
                return
        else:
            refData = Data()
            try:
                refData.set_from_trc(self.refPath)
            except:
                QMessageBox.critical(self, "Caution", "{0} cannot be loaded".format(self.refPath))
                return
            inpData = Data()
            try:
                inpData.set_from_trc(self.inpPath)
            except:
                QMessageBox.critical(self, "Caution", "{0} cannot be loaded".format(self.inpPath))
                return

        self.parent.reset()

        try:
            loadingDialog = LoadingDialog(self.contextsSet ,str(self.comboBoxCalculationType.currentText()), inpData, refData,
                                          fps=int(self.lineeditFps.text()), maximumGapTime=float(self.lineEditMaxGapTime.text()), parent=self)
            loadingDialog.setWindowModality(Qt.ApplicationModal)
            loadingDialog.show()
            #colors = loadingDialog.start(inpData, refData, fps=int(self.lineeditFps.text()), maximumGapTime=float(self.lineEditMaxGapTime.text()))
            """
            sys.stdout = Logger(self)
            
            DP_ = dp.DP(refData, inpData, verbose=True, ignoreWarning=True, verboseNan=True)
            colors = DP_.resultVisualization(fps=int(self.lineeditFps.text()),
                                         maximumGapTime=float(self.lineEditMaxGapTime.text()))

            self.parent.initialDraw(inpData, colors)
            """
        except Exception as e:
            tb = sys.exc_info()[2]
            QMessageBox.critical(self, "Caution", "Unexpected error occured:{0}".format(e.with_traceback(tb)))

    def manuallyReadContextsClicked(self):
        basedir = ''
        filters = "CONTEXTS files(*.contexts)"
        if self.comboBoxSkeltonType.currentText() == 'Volleyball':
            if self.parent.caches['prevContextsDirVolleyball'] is not None:
                basedir = self.parent.caches['prevContextsDirVolleyball']
        elif self.comboBoxSkeltonType.currentText() == 'Baseball':
            if self.parent.caches['prevContextsDirBaseball'] is not None:
                basedir = self.parent.caches['prevContextsDirBaseball']

        contextsPath, __ = QFileDialog.getOpenFileName(self, 'load file', basedir, filters)
        # check whether contexts file is valid or not
        def checkContextFile():
            if contextsPath == "":
                return None
            try:
                with open(contextsPath, 'r') as f:
                    contexts_ = []
                    lines = f.readlines()
                    for line in lines:
                        values = line.strip().split(',')
                        if values[-1] == "":
                            values = values[:-1]
                        contexts_.append(values)
                    return contexts_
            except Exception as e:
                tb = sys.exc_info()[2]
                QMessageBox.critical(self, "Caution", "\"{0}\" is invelid file\n{1}".format(contextsPath, e.with_traceback(tb)))

        checkResult = checkContextFile()
        if checkResult is not None:
            self.contextsSet['type'] = contextsPath
            self.contextsSet['contexts'] = checkResult
            self.labelContextsType.setText(os.path.basename(contextsPath))
            exec('self.parent.caches[\'prevContextsDir{0}\'] = os.path.dirname(contextsPath)'.format(
                self.comboBoxSkeltonType.currentText()))
        else:
            self.contextsSet['type'] = 'Default'
            self.labelContextsType.setText(self.contextsSet['type'])

    def play(self):
        self.buttonPlay.setEnabled(False)
        self.buttonPause.setEnabled(True)
        self.timer.start(int(1000.0 / int(self.lineeditFps.text())))  # ms

    def pause(self):
        self.buttonPlay.setEnabled(True)
        self.buttonPause.setEnabled(False)
        self.timer.stop()

    def updateVideo(self):
        if self.parent.frame < self.parent.frame_max:
            self.parent.frame += 1
            self.parent.sliderSetValue(self.parent.frame)
            self.parent.draw()
        else:
            self.pause()


class Calculator(QThread):
    finSignal = pyqtSignal(object)

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.parent = parent


    def run(self):
        try:
            sys.stdout = Logger(self.parent)
            kwargs = {'reference': self.parent.refData, 'input': self.parent.inpData,
                      'verbose': True, 'ignoreWarning': True, 'verboseNan': True}
            if self.parent.calcType == 'Independent':
                DP_ = DP(**kwargs)
            elif self.parent.calcType == 'Synchronous Contexts':
                DP_ = SyncContextDP(self.parent.contextsSet['contexts'], **kwargs)
            elif self.parent.calcType == 'Asynchronous Contexts':
                DP_ = AsyncContextDP(self.parent.contextsSet['contexts'], **kwargs)
            colors = DP_.resultVisualization(fps=self.parent.fps, maximumGapTime=self.parent.maximumGapTime)

            sys.stdout = sys.__stdout__

            self.finSignal.emit(colors)
        except Exception as e:
            tb = sys.exc_info()[2]
            self.finSignal.emit([e, tb])

class Logger(object):

    def __init__(self, parent):
        self.parent = parent


    def write(self, log):
        self.parent.textEdit.setText(log)
        """
        cursor = self.parent.textEdit.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(log)
        self.parent.textEdit.setTextCursor(cursor)
        self.parent.textEdit.ensureCursorVisible()
        """

    def flush(self):
        pass

class LoadingDialog(QMainWindow):
    def __init__(self, contextsSet, calcType, inpData, refData, fps, maximumGapTime, parent=None):
        QMainWindow.__init__(self, parent)
        self.parent = parent

        self.calcType = calcType
        self.contextsSet = contextsSet
        self.inpData = inpData
        self.refData = refData
        self.fps = fps
        self.maximumGapTime = maximumGapTime

        self.initUI()

    def initUI(self):
        self.mainWidget = QWidget(self)

        hbox = QHBoxLayout()

        self.movie_screen = QLabel()
        gifpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "calculating.gif")
        self.movie = QMovie(gifpath, QByteArray(), self)
        self.movie.setCacheMode(QMovie.CacheAll)
        self.movie.setSpeed(100)
        self.movie_screen.setMovie(self.movie)
        self.movie.start()
        hbox.addWidget(self.movie_screen)

        #self.textEdit = QTextEdit()
        self.textEdit = QLabel()
        hbox.addWidget(self.textEdit)

        self.mainWidget.setLayout(hbox)
        self.setCentralWidget(self.mainWidget)

        self.calculator = Calculator(self)
        self.calculator.finSignal.connect(self.finished)
        self.calculator.start()


    def finished(self, colors):
        self.parent.parent.initialDraw(self.inpData, colors)
        self.close()

class PreferenceDialog(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.parent = parent

        self.initUI()

    def initUI(self):
        self.mainWidget = QWidget(self)

        hbox = QHBoxLayout()

        self.movie_screen = QLabel()
        gifpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "calculating.gif")
        self.movie = QMovie(gifpath, QByteArray(), self)
        self.movie.setCacheMode(QMovie.CacheAll)
        self.movie.setSpeed(100)
        self.movie_screen.setMovie(self.movie)
        self.movie.start()
        hbox.addWidget(self.movie_screen)

        #self.textEdit = QTextEdit()
        self.textEdit = QLabel()
        hbox.addWidget(self.textEdit)

        self.mainWidget.setLayout(hbox)
        self.setCentralWidget(self.mainWidget)

    def closeEvent(self, a0: QCloseEvent):
        pass