from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys, os

class Calculator(QThread):
    finSignal = pyqtSignal(object)

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.parent = parent


    def run(self):
        try:
            sys.stdout = Logger(self.parent)
            # implement
            colors = self.parent.DP.resultVisualization(**self.parent.implementKwargs)

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
    def __init__(self, DP, implementKwargs, parent=None):
        QMainWindow.__init__(self, parent)
        self.parent = parent

        self.DP = DP
        if not isinstance(implementKwargs, dict):
            raise ValueError('\'implementKwargs\' must be dict, but got {0}'.format(type(implementKwargs).__name__))
        self.implementKwargs = implementKwargs

        self.initUI()

    def initUI(self):
        self.mainWidget = QWidget(self)

        hbox = QHBoxLayout()

        self.movie_screen = QLabel()
        gifpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "icon", "calculating.gif")
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
        inpData = self.DP.input
        self.parent.parent.initialDraw(inpData, colors)
        self.close()
