from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import os

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