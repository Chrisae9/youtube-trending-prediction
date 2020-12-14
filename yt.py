import logging
import os
import sys
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QIcon, QKeySequence
from PyQt5.QtWidgets import QApplication, QLineEdit, QMessageBox, QShortcut
from utils import compute

def get_ui_file_for(ui_filename):
    """
    Finds the path for the specified UI file and returns it.

    Checks to see if the applcation is an executable. 
    (sys._MEIPASS is a temporary directory for PyInstaller)
    """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, ui_filename)

    return os.path.join(os.path.dirname(os.path.realpath(__file__)), ui_filename)


UI_MainWindow, QtBaseClass = uic.loadUiType(get_ui_file_for('yt.ui'))


class YTApp(UI_MainWindow, QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        QtWidgets.QMainWindow.__init__(self)
        UI_MainWindow.__init__(self)
        self.setupUi(self)
        self.setWindowTitle('YT Predictor')
        self.run_button.clicked.connect(self.run)
        self.quit_button.clicked.connect(self.quit)
        self.quit_button.shortcut = QShortcut(QKeySequence('CTRL+Q'), self)
        self.quit_button.shortcut.activated.connect(self.quit)

        self.update_status()

        self.categories = {
            "Film & Animation": 1,
            "Autos & Vehicles": 2,
            "Music": 10,
            "Pets & Animals": 15,
            "Sports": 17,
            "Short Movies": 18,
            "Travel & Events": 19,
            "Gaming": 20,
            "Videoblogging": 21,
            "People & Blogs": 22,
            "Comedy": 34,
            "Entertainment": 24,
            "News & Politics": 25,
            "Howto & Style": 26,
            "Education": 27,
            "Science & Technology": 28,
            "Movies": 30,
            "Anime/Animation": 31,
            "Action/Adventure": 32,
            "Classics": 33,
            "Documentary": 35,
            "Drama": 36,
            "Family": 37,
            "Foreign": 38,
            "Horror": 39,
            "Sci-Fi/Fantasy": 40,
            "Thriller": 41,
            "Shorts": 42,
            "Shows": 43,
            "Trailers": 44
        }

        self.tod = {
            'Night': 0,
            'Morning': 1,
            'Afternoon': 2,
            'Evening': 3
        }

        for x in list(self.categories.keys()):
            self.cate_combo.addItem(x)

        for x in list(self.tod.keys()):
            self.time_combo.addItem(x)

    def get_data(self):
        """gets the data"""
        return compute(self.title_input.text(), self.descr_box.toPlainText(), self.tags_input.toPlainText().split(
            ','), self.tod[self.time_combo.currentText()], self.categories[self.cate_combo.currentText()])

    def data_empty(self):
        """Returns true if the data is empty"""
        if len(self.title_input.text()) and len(self.descr_box.toPlainText()) and len(self.tags_input.toPlainText()):
            return False
        return True

    def clear_data(self):
        """Clears the username and password field"""
        self.title_input.clear()
        self.descr_box.clear()
        self.tags_input.clear()

    def toggle_data(self, toggle=True):
        """Toggles the username and password box"""
        if toggle:
            self.title_input.setEnabled(True)
            self.descr_box.setEnabled(True)
            self.tags_input.setEnabled(True)
            self.cate_combo.setEnabled(True)
            self.time_combo.setEnabled(True)

        else:
            self.title_input.setEnabled(False)
            self.descr_box.setEnabled(False)
            self.tags_input.setEnabled(False)
            self.cate_combo.setEnabled(False)
            self.time_combo.setEnabled(False)

    def run(self):
        """runs the script"""
        value = 0
        print("here")

        if self.data_empty():
            self.update_status('Please enter values')

        else:
            self.toggle_data(False)
            self.update_status('Processing')

            # PUT PROCESSING HERE

            answer = self.get_data()
            if(answer == 1):
                answer = '<html><head/><body><p align="center"> There is a high chance <br/>your video will go trending within 24 hours.</p></body></html>'
            else:
                answer = '<html><head/><body><p align="center"> There is a low chance <br/>your video will go trending within 24 hours.</p></body></html>'

            self.clear_data()
            self.toggle_data()
            self.update_status('Done')
            QMessageBox.about(self, 'Answer', answer)

    def update_status(self, text='Ready'):
        """Updates the status bar text at the bottom of the app"""
        self.statusBar().showMessage(text)

    def quit(self):
        """Quits the app"""
        QApplication.quit()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = YTApp()
    window.show()
    logging.debug('Main window shown')
    app.exec_()


if __name__ == '__main__':
    main()
