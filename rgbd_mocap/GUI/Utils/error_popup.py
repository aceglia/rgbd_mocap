import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class ErrorPopUp(QMessageBox):
    """
        Pop up showing the 'error_message'
    """

    def __init__(self, error_message: str = "An un-excepted error happened", parent=None):
        super(ErrorPopUp, self).__init__(parent)
        self.setText(error_message)
        self.setWindowTitle('Error !')
        self.exec()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = QMainWindow()
    button_error = QPushButton(main_window)
    main_window.setCentralWidget(button_error)

    button_error.clicked.connect(lambda: ErrorPopUp())

    main_window.show()
    sys.exit(app.exec_())
