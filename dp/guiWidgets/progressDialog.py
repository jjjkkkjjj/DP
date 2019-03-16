# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys

class Implement(QThread):
    valueChanged = pyqtSignal(object)
    finSignal = pyqtSignal(object)

    def __init__(self, parent=None):
        super(Implement, self).__init__(parent)
        self.flag = False

    def setValue(self, value, appendedText):
        self.valueChanged.emit({'value':value, 'appendedText':appendedText})

    def finish(self):
        self.finSignal.emit([])

    def abort(self, text):
        self.finSignal.emit([text])

class ProgressBar(QDialog):
    def __init__(self, job, parent, closeDialogShow=True, closeDialogComment=''):
        super().__init__(parent)

        if not issubclass(job.__class__, Implement):
            raise TypeError('job must be inherenced \'Implement\'')
        self.job = job
        self.parent = parent
        self.closeDialogShow = closeDialogShow
        self.closeDialogComment = closeDialogComment
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Progress')

        vbox = QVBoxLayout()
        hboxprogress = QHBoxLayout()

        self.progress = QProgressBar(self.parent)
        self.progress.setMaximum(100)
        hboxprogress.addWidget(self.progress)
        self.labelProgress = QLabel('0%')
        hboxprogress.addWidget(self.labelProgress)
        vbox.addLayout(hboxprogress)

        self.setLayout(vbox)
        self.setWindowModality(Qt.ApplicationModal)
        self.show()

    def run(self):
        self.job.valueChanged.connect(self.setProgressValue)
        self.job.finSignal.connect(self.finish)
        self.job.flag = True
        self.job.start()

    def setProgressValue(self, result):
        value = result.pop('value')
        appendedText = result.pop('appendedText', '')
        self.progress.setValue(value)
        self.labelProgress.setText(str(value) + '%' + appendedText)
        QApplication.processEvents()

    def closeEvent(self, event):
        self.job.flag = False

    def finish(self, result):
        if len(result) == 0:
            self.close()
            if self.closeDialogShow:
                QMessageBox.information(self.parent, "Finished", self.closeDialogComment, QMessageBox.Ok)
        elif len(result) == 1:
            self.close()
            QMessageBox.information(self.parent, "Finished", result[0], QMessageBox.Ok)
        else:
            self.close()
            e = result[0]
            tb = sys.exc_info()[1]
            QMessageBox.critical(self.parent, "Error", "Unexpencted error was occurred: {0}".format(e.with_traceback(tb)), QMessageBox.Ok)
