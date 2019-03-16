from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys, os
from .calculator import LoadingDialog

class LeftDockWidget(QWidget):
    def __init__(self, dpModule, parent):
        QWidget.__init__(self, parent)
        self.parent = parent

        self.dpModule = dpModule
        self.contextsSet = {'type': 'Default', 'contexts': self.dpModule['contexts']('Baseball')}
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

        hboxRefButton = QHBoxLayout()
        self.buttonOpenRefPath = QPushButton("Open")
        self.buttonOpenRefPath.clicked.connect(lambda: self.openClicked(True))
        hboxRefButton.addWidget(self.buttonOpenRefPath)

        self.buttonViewRefPath = QPushButton("View")
        self.buttonViewRefPath.clicked.connect(lambda: self.viewClicked(True))
        self.buttonViewRefPath.setEnabled(False)
        hboxRefButton.addWidget(self.buttonViewRefPath)
        vboxConfig.addLayout(hboxRefButton)

        self.labelRefRange = QLabel()
        vboxConfig.addWidget(self.labelRefRange)

        self.labelInputPatternPath = QLabel()
        self.labelInputPatternPath.setText("Input Pattern Path")
        vboxConfig.addWidget(self.labelInputPatternPath)

        self.labelRealInpPath = QLabel()
        self.labelRealInpPath.setText("No selected")
        vboxConfig.addWidget(self.labelRealInpPath)

        hboxInpButton = QHBoxLayout()
        self.buttonOpenInpPath = QPushButton("Open")
        self.buttonOpenInpPath.clicked.connect(lambda: self.openClicked(False))
        hboxInpButton.addWidget(self.buttonOpenInpPath)

        self.buttonViewInpPath = QPushButton("View")
        self.buttonViewInpPath.clicked.connect(lambda: self.viewClicked(False))
        self.buttonViewInpPath.setEnabled(False)
        hboxInpButton.addWidget(self.buttonViewInpPath)
        vboxConfig.addLayout(hboxInpButton)

        self.labelInpRange = QLabel()
        vboxConfig.addWidget(self.labelInpRange)

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
        self.buttonViewRefPath.setEnabled(False)
        self.buttonViewInpPath.setEnabled(False)

        self.contextsSet = {'type': 'Default', 'contexts': self.dpModule['contexts'](str(self.comboBoxSkeltonType.currentText()))}
        self.labelContextsType.setText(self.contextsSet['type'])

    def openClicked(self, ref):
        basedir = ''
        if self.comboBoxSkeltonType.currentText() == 'Volleyball':
            filters = "TRC files(*.trc)"
            if self.parent.caches['prevDirVolleyball'] is not None:
                basedir = self.parent.caches['prevDirVolleyball']
        elif self.comboBoxSkeltonType.currentText() == 'Baseball':
            filters = "CSV files(*.csv);;TRC files(*.trc)"
            if self.parent.caches['prevDirBaseball'] is not None:
                basedir = self.parent.caches['prevDirBaseball']
        else:
            filters = "TRC files(*.trc)"

        if ref:
            refPath, __ = QFileDialog.getOpenFileName(self, 'load file', basedir, filters)
            if refPath != "" and self._checkData(True, refPath):
                self.refPath = refPath
                self.labelRealRefPath.setText(os.path.basename(self.refPath))
                exec('self.parent.caches[\'prevDir{0}\'] = os.path.dirname(self.refPath)'.format(
                    self.comboBoxSkeltonType.currentText()))
                self.buttonViewRefPath.setEnabled(str(self.comboBoxSkeltonType.currentText()) == 'Baseball')
            else:
                self.refPath = None
                self.labelRealRefPath.setText("No selected")
                self.buttonViewRefPath.setEnabled(False)

        else:
            inpPath, __ = QFileDialog.getOpenFileName(self, 'load file', basedir, filters)
            if inpPath != "" and self._checkData(False, inpPath):
                self.inpPath = inpPath
                self.labelRealInpPath.setText(os.path.basename(self.inpPath))
                exec('self.parent.caches[\'prevDir{0}\'] = os.path.dirname(self.inpPath)'.format(
                    self.comboBoxSkeltonType.currentText()))
                self.buttonViewInpPath.setEnabled(str(self.comboBoxSkeltonType.currentText()) == 'Baseball')
            else:
                self.inpPath = None
                self.labelRealInpPath.setText("No selected")
                self.buttonViewInpPath.setEnabled(False)

        if self.refPath is not None and self.inpPath is not None:
            self.buttonDone.setEnabled(True)
        else:
            self.buttonDone.setEnabled(False)

    def viewClicked(self, ref):
        if ref:
            qlabel = self.labelRefRange
            data = self.dpModule['csvReader'](os.path.basename(self.refPath), os.path.dirname(self.refPath))
        else:
            qlabel = self.labelInpRange
            data = self.dpModule['csvReader'](os.path.basename(self.inpPath), os.path.dirname(self.inpPath))
        self.parent.showNormalViewer(qlabel, data)


    def doneClicked(self):
        if self.comboBoxSkeltonType.currentText() == 'Baseball' and os.path.basename(self.refPath)[-4:] in ['.csv', '.CSV']:
            refData = self.dpModule['csvReader'](os.path.basename(self.refPath), os.path.dirname(self.refPath))
            # rename joint
            for joint in list(refData.joints.keys()):
                refData.joints[joint.split(':')[-1]] = refData.joints.pop(joint)
        else:
            refData = self.dpModule['Data']()
            refData.set_from_trc(self.refPath)
        if self.comboBoxSkeltonType.currentText() == 'Baseball' and os.path.basename(self.inpPath)[-4:] in ['.csv', '.CSV']:
            inpData = self.dpModule['csvReader'](os.path.basename(self.inpPath), os.path.dirname(self.inpPath))
            # rename joint
            for joint in list(inpData.joints.keys()):
                inpData.joints[joint.split(':')[-1]] = inpData.joints.pop(joint)
        else:
            inpData = self.dpModule['Data']()
            inpData.set_from_trc(self.inpPath)
        # cut
        ini, fin = str(self.labelRefRange.text()).split('~')
        refData.cutFrame(ini=int(ini), fin=int(fin), save=True, update=False)
        ini, fin = str(self.labelInpRange.text()).split('~')
        inpData.cutFrame(ini=int(ini), fin=int(fin), save=True, update=False)

        self.parent.reset()

        try:
            kwargs = {'reference': refData, 'input': inpData,
                      'verbose': True, 'ignoreWarning': True, 'verboseNan': True}

            fps = int(self.lineeditFps.text())
            maximumGapTime = float(self.lineEditMaxGapTime.text())
            #implementKwargs = {'kind': 'visualization2', 'fps': fps, 'maximumGapTime': maximumGapTime}
            implementKwargs = {'kind': 'localdiff', 'fps': fps, 'maximumGapTime': maximumGapTime}
            calcType = str(self.comboBoxCalculationType.currentText())
            if calcType == 'Independent':
                DP_ = self.dpModule['DP'](**kwargs)
            elif calcType == 'Synchronous Contexts':
                DP_ = self.dpModule['SyncContextDP'](self.contextsSet['contexts'], **kwargs)
            elif calcType == 'Asynchronous Contexts':
                DP_ = self.dpModule['ASyncContextDP'](self.contextsSet['contexts'], **kwargs)
                # add kinds
                kinds = []
                for context in self.contextsSet['contexts']:
                    #kinds.append('async{0}-visualization2'.format(len(context)))
                    kinds.append('async{0}-localdiff'.format(len(context)))
                implementKwargs['kinds'] = kinds

            else:
                raise NameError("{0} is invalid calculation type".format(calcType))

            loadingDialog = LoadingDialog(DP=DP_, implementKwargs=implementKwargs, parent=self)
            loadingDialog.setWindowModality(Qt.ApplicationModal)
            loadingDialog.show()
            # colors = loadingDialog.start(inpData, refData, fps=int(self.lineeditFps.text()), maximumGapTime=float(self.lineEditMaxGapTime.text()))
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
                QMessageBox.critical(self, "Caution",
                                     "\"{0}\" is invelid file\n{1}".format(contextsPath, e.with_traceback(tb)))

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

    def _checkData(self, ref, path):
        if self.comboBoxSkeltonType.currentText() == 'Baseball' and os.path.basename(path)[-4:] in ['.csv', '.CSV']:
            if ref:
                refData = self.dpModule['csvReader'](os.path.basename(path), os.path.dirname(path))
                if refData is None:
                    QMessageBox.critical(self, "Caution", "{0} cannot be loaded".format(path))
                    self.labelRefRange.setText('')
                    return False
                self.labelRefRange.setText('0~{0}'.format(refData.frame_max - 1))
            else:
                inpData = self.dpModule['csvReader'](os.path.basename(path), os.path.dirname(path))
                if inpData is None:
                    QMessageBox.critical(self, "Caution", "{0} cannot be loaded".format(path))
                    self.labelInpRange.setText('')
                    return False
                self.labelInpRange.setText('0~{0}'.format(inpData.frame_max - 1))
        else:
            if ref:
                refData = self.dpModule['Data']()
                try:
                    refData.set_from_trc(path)
                except:
                    QMessageBox.critical(self, "Caution", "{0} cannot be loaded".format(path))
                    self.labelRefRange.setText('')
                    return False
                self.labelRefRange.setText('0~{0}'.format(refData.frame_max - 1))
            else:
                inpData = self.dpModule['Data']()
                try:
                    inpData.set_from_trc(path)
                except:
                    QMessageBox.critical(self, "Caution", "{0} cannot be loaded".format(path))
                    self.labelInpRange.setText('')
                    return False
                self.labelInpRange.setText('0~{0}'.format(inpData.frame_max - 1))
        return True