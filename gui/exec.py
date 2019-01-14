from dp.gui import DPgui
import sys
from PyQt5.QtWidgets import *

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = DPgui()
    gui.show()
    sys.exit(app.exec_())