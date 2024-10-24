from PyQt5.QtWidgets import *
import sys
from rgbd_mocap.GUI.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    demo = MainWindow()
    # demo.vce.dir = "../../../../rgbd_mocap/data_files/P4_session2/gear_20_15-08-2023_10_52_14/"
    # demo.vce.load_images()
    demo.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
