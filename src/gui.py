from PyQt5.QtWidgets import (QMainWindow,
                             QPushButton,
                             QFrame, QFormLayout, QPlainTextEdit, QFileDialog, QLabel,
                             QApplication
                             )

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QFont, QPixmap

import sys
from OCR.ocr_mp import *

class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'Simple OCR'
        self.filename = None
        self.chars_pre_path = "../data/zad2_silmarillion/chars/"

        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(350, 150, 1200, 800)
        self.setLayout(QFormLayout())
        self.resizeEvent = self.resize_


        # Create a choose image button
        self.button_ch = QPushButton('choose file', self)
        self.button_ch.setFont(QFont("Times", 12))
        self.button_ch.setGeometry(20, 125, 120, 40)
        self.button_ch.clicked.connect(self.on_click_choose)

        # Create a analyze image button
        self.button_an = QPushButton('analyze file', self)
        self.button_an.setFont(QFont("Times", 12))
        self.button_an.setGeometry(20, 170, 120, 40)
        self.button_an.clicked.connect(self.on_click_analyze_text)

        self.choosen_file = QPlainTextEdit(self.filename, self)
        self.choosen_file.setGeometry(20, 40, 500, 70)
        self.choosen_file.setFont(QFont("Times", 12))
        self.choosen_file.setReadOnly(True)

        self.parsed_text = QPlainTextEdit(self.filename, self)
        self.parsed_text.setGeometry(20, 210, 500, 600)
        self.parsed_text.setFont(QFont("Times", 12))
        self.parsed_text.setReadOnly(True)



        # im text
        frm = QFrame(self)
        self.image = QPixmap()

        self.label = QLabel()
        self.label.setGeometry(600, 40, 500, 700)
        self.layout().addWidget(self.label)


        self.show()

    def on_click_choose(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            self.filename = fileName
            self.choosen_file.setPlainText(self.filename)
            self.image.load(self.filename)
            self.label.clear()

            print(self.filename)
            wi, hi = self.image.width(), self.image.height()
            w = self.label.width()
            h = self.label.height()
            scal = max(wi / w, hi / h)
            self.image = self.image.scaled(wi / scal, hi / scal)
            self.label.setPixmap(self.image)
            self.parsed_text.setPlainText("")


    @pyqtSlot()
    def on_click_analyze_text(self):
        if self.filename is not None:
            text = main_ocr(self.filename, self.chars_pre_path)
            self.parsed_text.setPlainText(text)
            plt.show()



    @pyqtSlot()
    def resize_(self, event):
        w = self.width()
        h = self.height()
        self.parsed_text.resize(w / 2 - 100, h - 280)
        self.label.setGeometry(20 + w/2, 40, w/2 - 100, h - 200)
        if self.image is not None and self.image.width() > 0:
            wi, hi = self.image.width(), self.image.height()
            w = self.label.width()
            h = self.label.height()
            scal = max(wi / w, hi / h)
            self.image.load(self.filename)
            self.image = self.image.scaled(wi / scal, hi / scal)
            self.label.setPixmap(self.image)
