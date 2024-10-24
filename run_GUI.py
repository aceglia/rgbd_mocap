from PyQt5.QtWidgets import *
import sys
from rgbd_mocap.GUI.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    demo = MainWindow()
    demo.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
