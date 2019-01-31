from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import os

class PreferenceDialog(QMainWindow):
    def __init__(self, constraint, parent=None):
        QMainWindow.__init__(self, parent)
        self.parent = parent

        self.constraint = constraint
        self.calcCheck = {}
        self.initUI()

    def initUI(self):
        self.mainWidget = QWidget(self)

        vbox = QVBoxLayout()

        # independent
        self.groupIndependent = QGroupBox("Independent")
        vboxIndependent = QVBoxLayout()
        hboxIndependent = QHBoxLayout()

        self.labelIndCalcType = QLabel()
        self.labelIndCalcType.setText("Calculation Type")
        hboxIndependent.addWidget(self.labelIndCalcType)

        self.lineeditIndCalcType = QLineEdit()
        self.lineeditIndCalcType.setText("aaaa")
        self.calcCheck['independet'] = False
        hboxIndependent.addWidget(self.lineeditIndCalcType)

        self.qlabelIndIcon = QLabel()

        hboxIndependent.addWidget(self.qlabelIndIcon)
        self.lineeditIndCalcType.textChanged.connect(
            lambda: self._checkConstraint(self.lineeditIndCalcType, self.qlabelIndIcon, 'independent'))

        self.groupIndependent.setLayout(hboxIndependent)

        # syncContext
        self.groupSyncContext = QGroupBox("SyncContext")
        vboxSyncContext = QVBoxLayout()
        hboxSyncContext1 = QHBoxLayout()

        self.labelSyncContextCalcType1 = QLabel()
        self.labelSyncContextCalcType1.setText("Calculation Type")
        hboxSyncContext1.addWidget(self.labelSyncContextCalcType1)

        self.lineeditSyncContextCalcType1 = QLineEdit()
        self.lineeditSyncContextCalcType1.setText("aaaa")
        self.calcCheck['syncContext1'] = False
        hboxSyncContext1.addWidget(self.lineeditSyncContextCalcType1)

        self.qlabelSyncContextIcon1 = QLabel()

        hboxSyncContext1.addWidget(self.qlabelSyncContextIcon1)
        self.lineeditSyncContextCalcType1.textChanged.connect(
            lambda: self._checkConstraint(self.lineeditSyncContextCalcType1, self.qlabelSyncContextIcon1, 'syncContext1'))

        hboxSyncContext2 = QHBoxLayout()

        self.labelSyncContextCalcType2 = QLabel()
        self.labelSyncContextCalcType2.setText("Calculation Type")
        hboxSyncContext2.addWidget(self.labelSyncContextCalcType2)

        self.lineeditSyncContextCalcType2 = QLineEdit()
        self.lineeditSyncContextCalcType2.setText("aaaa")
        self.calcCheck['syncContext2'] = False
        hboxSyncContext2.addWidget(self.lineeditSyncContextCalcType2)

        self.qlabelSyncContextIcon2 = QLabel()

        hboxSyncContext2.addWidget(self.qlabelSyncContextIcon2)
        self.lineeditSyncContextCalcType2.textChanged.connect(
            lambda: self._checkConstraint(self.lineeditSyncContextCalcType2, self.qlabelSyncContextIcon2, 'syncContext2'))

        vboxSyncContext.addLayout(hboxSyncContext1)
        vboxSyncContext.addLayout(hboxSyncContext2)
        self.groupSyncContext.setLayout(vboxSyncContext)

        # asyncContext
        self.groupASyncContext = QGroupBox("ASyncContext")
        vboxASyncContext = QVBoxLayout()
        hboxASyncContext1 = QHBoxLayout()

        self.labelASyncContextCalcType1 = QLabel()
        self.labelASyncContextCalcType1.setText("Calculation Type")
        hboxASyncContext1.addWidget(self.labelASyncContextCalcType1)

        self.lineeditASyncContextCalcType1 = QLineEdit()
        self.lineeditASyncContextCalcType1.setText("aaaa")
        self.calcCheck['ASyncContext1'] = False
        hboxASyncContext1.addWidget(self.lineeditASyncContextCalcType1)

        self.qlabelASyncContextIcon1 = QLabel()

        hboxASyncContext1.addWidget(self.qlabelASyncContextIcon1)
        self.lineeditASyncContextCalcType1.textChanged.connect(
            lambda: self._checkConstraint(self.lineeditASyncContextCalcType1, self.qlabelASyncContextIcon1,
                                          'ASyncContext1'))

        hboxASyncContext2 = QHBoxLayout()

        self.labelASyncContextCalcType2 = QLabel()
        self.labelASyncContextCalcType2.setText("Calculation Type")
        hboxASyncContext2.addWidget(self.labelASyncContextCalcType2)

        self.lineeditASyncContextCalcType2 = QLineEdit()
        self.lineeditASyncContextCalcType2.setText("aaaa")
        self.calcCheck['ASyncContext2'] = False
        hboxASyncContext2.addWidget(self.lineeditASyncContextCalcType2)

        self.qlabelASyncContextIcon2 = QLabel()

        hboxASyncContext2.addWidget(self.qlabelASyncContextIcon2)
        self.lineeditASyncContextCalcType2.textChanged.connect(
            lambda: self._checkConstraint(self.lineeditASyncContextCalcType2, self.qlabelASyncContextIcon2,
                                          'ASyncContext2'))

        vboxASyncContext.addLayout(hboxASyncContext1)
        vboxASyncContext.addLayout(hboxASyncContext2)
        self.groupASyncContext.setLayout(vboxASyncContext)

        vbox.addWidget(self.groupIndependent)
        vbox.addWidget(self.groupSyncContext)
        vbox.addWidget(self.groupASyncContext)

        self.mainWidget.setLayout(vbox)
        self.setCentralWidget(self.mainWidget)

    def _checkConstraint(self, qlineedit, qlabel_icon, name):
        kind = str(qlineedit.text())
        try:
            if 'visualization' in kind:
                raise NameError
            self.constraint(kind)
        except NameError:
            iconpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "icon", "caution.png")
            qlabel_icon.setPixmap(QPixmap(iconpath))
            qlabel_icon.show()
            self.calcCheck[name] = False
            return

        iconpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "icon", "check.png")
        qlabel_icon.setPixmap(QPixmap(iconpath))
        qlabel_icon.show()
        self.calcCheck[name] = True
        return

    def closeEvent(self, a0: QCloseEvent):
        pass



def readPreference():
    pass

def writePreference():
    pass