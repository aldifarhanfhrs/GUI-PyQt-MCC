#Made by Aldi Farhan Fahrosa

import sys
import time
import joblib
import cv2 as cv
import numpy as np
from decimal import Decimal

import cv2
from PIL import Image, ImageDraw, ImageQt
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QApplication, QCheckBox, QMessageBox, QDialog

from PyQt5.uic import loadUi
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
import pyqtgraph as pg

#library agar camera tidak lag saat realtime
from PyQt5.QtCore import Qt, QThread, pyqtSignal

import io
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

class cameraThread(QThread):
    frameCaptured = pyqtSignal(np.ndarray)
    
    def __init__(self, camera_index):
        super(cameraThread, self).__init__()
        self.camera_index = camera_index
        self.running = False
        
    def run(self):
        self.running = False
        cap = cv2.VideoCapture(self.camera_index)
        
        while self.running:
            ret, frame = cap.read()
            if ret:
                self.frameCaptured.emit(frame)
                self.msleep(10)
                
        cap.release()
        
    def stop(self):
        self.running = False
        self.wait()
        
class IndividualCamera(QMainWindow):
    def __init__(self):
        super(IndividualCamera, self).__init__()
        loadUi(r"C:\Users\aldif\OneDrive\Documents\Semester VII\Python\PyQt\MCC\Individual Inspection\individualInspection.ui", self)
        self.camera_thread = None
        
        self.chooseCameraBox.currentIndexChanged.connect(self.selectCamera)
        
        self.captureButton.clicked.connect(self.captureAndDisplayImage)
        #self.enableRealtimeCB.isChecked.connect(self.captureAndDisplayImage)
        
        self.capture_counter = 0

    def startCamera(self):
        if self.camera_thread is not None and self.enableRealtimeCB.isChecked() and self.camera_thread.isRunning():
            self.stop_camera()
    
    def selectCamera(self):            
        selectedCamera = self.chooseCameraBox.currentText()
        
        if selectedCamera == 'Camera Internal':
            self.camera_thread = cameraThread(0)
        elif selectedCamera == 'Camera External':
            self.camera_thread == cameraThread(1)
            
        self.camera_thread.frameCaptured.connect(self.updateImage)
        self.camera_thread.start()
        
    def stopCamera(self):
        if self.camera_thread is not None:
            self.camera_thead.stop()
    
    def captureAndDisplayImage(self):
        if self.camera_thread is not None and self.camera_thread.isRunning():
            ret, frame = self.camera_thread.framecaptured.emit(frame)
    

if __name__ == "__main__":
    app = QApplication([])
    window = IndividualCamera()
    window.show()
    app.exec_()