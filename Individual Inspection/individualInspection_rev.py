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
from skimage.feature import greycomatrix, greycoprops

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

class individualCamera(QMainWindow):
    def __init__(self):
        global crop_coordinates
        super().__init__()
        loadUi(r"C:\Users\aldif\OneDrive\Documents\Semester VII\Python\PyQt\MCC\Individual Inspection\individualInspection.ui", self)
        #self.setWindowState(QtCore.Qt.WindowMaximized)
        
        self.available_cameras = QCameraInfo.availableCameras()
        if not self.available_cameras:
            sys.exit()
        
        self.setWindowTitle('MCC Individual Camera App')
        self.setWindowIcon(QIcon(r'C:\Users\aldif\OneDrive\Documents\Semester VII\Python\PyQt\MCC\Individual Inspection\logo MCC.png'))
        
        self.chooseCameraBox.currentIndexChanged.connect(self.selectCamera)
        self.enableRealtimeCB.stateChanged.connect(self.toggleCamera)
        self.enableSaveCB.stateChanged.connect(self.toggleSaveImage)
    
        #self.comboBox.currentIndexChanged.connect(self.olah_citra)

        self.captureButton.clicked.connect(self.captureAndDisplayImage)
        self.captureButton.clicked.connect(self.olah_citra)

        self.browseButton.clicked.connect(self.input_manual)
        self.browseButton.clicked.connect(self.olah_citra)
        
        self.capture_counter = 0
        
        
        self.timer = QTimer(self)
        self.camera = cv2.VideoCapture(0)
        self.timer.timeout.connect(self.updateFrame)
        self.is_camera_active = False
        self.save_image_enabled = False
        self.start_time = time.time()
        
        self.frame_count = 0
        #self.start_time = cv2.getTickCount() / cv2.getTickFrequency()
        
    
    def toggleCamera(self, state):
        self.is_camera_active = state == Qt.Checked
        if self.is_camera_active:
            self.timer.start(30)
        else:
            self.timer.stop()
    
    def toggleSaveImage(self, state):
        self.save_image_enabled = state == Qt.Checked
        
    def setCameraResolution(self, width, height):
        if self.camera is not None:
            self.camera.set(cv.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        
    def selectCamera(self):
        selectedCamera = self.chooseCameraBox.currentText()
        
        #self.timer.timeout.connect(self.updateFrame)
        #self.timer = QtCore.QTimer(self)
        #self.timer.timeout.connect(self.updateFrame)
        #self.timer.setInterval(10)
        
        if selectedCamera == 'Camera Internal':
            self.camera.release()
            self.camera = cv.VideoCapture(0)
            self.setCameraResolution(3264, 2448)
            self.camera.set(cv.CAP_PROP_FPS, 120)
            #fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            #self.timer.start()
        
        elif selectedCamera == 'Camera External':
            self.camera.release()
            self.camera = cv.VideoCapture(1)
            self.setCameraResolution(3264, 2448)
            self.camera.set(cv2.CAP_PROP_FPS, 120)
            #fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            #self.timer.start()
            
                
        elif selectedCamera == 'Choose Manually':
            # Stop the camera capture and take a picture
            pass
            #self.timer.stop()         

    def input_manual(self):
        global fname, pixmap, img, crop_coordinates
        fname = QFileDialog.getOpenFileName(self, 'Open File', r"C:\Users\aldif\OneDrive\Documents\Semester VII\Python\PyQt\Capture Image")

        img = Image.open(fname[0])

        crop_coordinates = (1200, 1200, 2020, 1750)
        x, y, x2, y2 = crop_coordinates

        draw = ImageDraw.Draw(img)
        draw.rectangle([(x, y), (x2, y2)], outline=(255, 0, 0), width=2)

        pixmap_img = QPixmap.fromImage(QImage(img.tobytes(), img.width, img.height, img.width * 3, QImage.Format_RGB888))

        #Open image
        img = Image.open(fname[0])

        self.realtimeVideoDisplay.setPixmap(pixmap_img)
        self.realtimeVideoDisplay.setScaledContents(True)
        self.directoryLabel.setText(fname[0])

    def olah_citra(self):
        start_time = time.time()

        #fname = QFileDialog.getOpenFileName(self, 'Open File', r'F:\Project\Philip Vision Checking\Data Botol\reject')
        #img = Image.open(image_filename)
        
        #ganti ini untuk data manual
        #img = Image.open(fname[0])

        
        #crop_coordinates = (240, 200, 400, 280) #ini untuk live camera
        left, upper, right, lower = crop_coordinates
        cropped_image = img.crop((left, upper, right, lower))

        gray_image = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2GRAY)

        thresholded_image = gray_image.copy()
        threshold_value = 120
        thresholded_image[thresholded_image > threshold_value] = 255

        hasil1 = model_1(thresholded_image)
        hasil2 = model_2(thresholded_image)
        num_objects, corner_coordinates = model_3(thresholded_image)

        image = cv2.cvtColor(np.array(thresholded_image), cv2.COLOR_GRAY2BGR)
        image = Image.fromarray(image) 
        draw = ImageDraw.Draw(image)

        for (x1, y1), (x2, y2) in corner_coordinates:
            draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)

        if num_objects == 12 :
            result3_1 = 'No'
        else:
            result3_1 = 'Yes'

        if result3_1 == 'Yes' or hasil1 == 'Yes' or hasil2 == 'Yes':
            decision = "Rejected"
        else:
            decision = "Good"

        self.objectLabel.setText(str(num_objects))

        self.catPudarLabel.setText(hasil1)
        self.printKurangLabel.setText(hasil2)
        self.overPrintingLabel.setText(result3_1)
        self.decisionLabel.setText(decision)


        fig = Figure()

        ax = fig.add_subplot(111)
        ax.hist(thresholded_image.ravel(), bins=256, range=(0, 250), histtype='stepfilled')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')

        buffer = io.BytesIO()
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        fig.savefig(buffer, format='rgba')
        buffer.seek(0)

        pixmap_histogram = QPixmap.fromImage(QImage(buffer.read(), *canvas.get_width_height(), QImage.Format_RGBA8888))
        self.histogramLabel.setPixmap(pixmap_histogram)
        self.histogramLabel.setScaledContents(True)


        pixmap_img = QPixmap.fromImage(QImage(image.tobytes(), image.width, image.height, image.width * 3, QImage.Format_RGB888))
        #pixmap_thresholded = QPixmap.fromImage(QImage(thresholded_image.data, thresholded_image.shape[1], thresholded_image.shape[0], thresholded_image.shape[1], QImage.Format_Grayscale8))
        self.ROILabel.setPixmap(pixmap_img)
        self.ROILabel.setScaledContents(True)

        waktu = time.time() - start_time
        waktu = "{:.2f}".format(waktu)
        self.timeLabel.setText(str(waktu))
        

    def captureAndDisplayImage(self):
        global image_filename, img, crop_coordinates
        if self.camera is not None:
            ret, captured_image = self.camera.read()

            if ret and self.save_image_enabled:
                height, width, channel = captured_image.shape
                bytesPerLine = channel * width

                captured_image_rgb = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)

                image_live = QImage(captured_image_rgb.data, width, height, bytesPerLine, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(image_live)
                self.ROILabel.setPixmap(pixmap)
                self.ROILabel.setScaledContents(True)
                
                self.capture_counter += 1

                timestamp = time.strftime("%Y%m%d_%H%M%S")
                image_filename = fr"C:\Users\aldif\OneDrive\Documents\Semester VII\Python\PyQt\Capture Image\BottleCaptured_{timestamp}_{self.capture_counter}.jpg"
                cv2.imwrite(image_filename, captured_image)

                img = Image.open(image_filename)
                crop_coordinates = (1200, 1200, 2020, 1750)

            #else:
                #self.statusBar().showMessage("Failed to capture an image.")
    
    def updateFrame(self):
        ret, frame = self.camera.read()
        if ret:
            # Convert the OpenCV image to a format suitable for QPixmap
            height, width, channel = frame.shape
            bytes_per_line = channel * width
            
            frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            q_image = QImage(frameRGB.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.realtimeVideoDisplay.setPixmap(pixmap.scaled(self.realtimeVideoDisplay.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)) 
            
            self.frame_count += 1

            if self.frame_count % 20 == 0:
                end_time = time.time()
                elapsed_time = end_time - self.start_time
                self.fps = 100.0 / elapsed_time
                self.fpsLabel.setText(f"{self.fps:.0f}")
                self.start_time = end_time        

#Model        
def model_1(image):
    pca_model_1 = r'C:\Users\aldif\OneDrive\Documents\Semester VII\Python\PyQt\MCC\Inspection\pca_model_1_normalized_histogram.pkl'
    classifier_model_1 = r'C:\Users\aldif\OneDrive\Documents\Semester VII\Python\PyQt\MCC\Inspection\model_1_RF_normalized_histogram.h5'
    
    pca = joblib.load(pca_model_1)
    rf_classifier = joblib.load(classifier_model_1)
    
    #hist, bins = np.histogram(image, bins=256, range=(0, 256))
    hist, bins = np.histogram(image, bins=256, range=(0, 256), density=True)

    X = hist[20:100]
    X = X.reshape(1, -1)
    
    X_transformed = pca.transform(X)
    
    X_transformed_subset = X_transformed[:, 0:3]
    predictions = rf_classifier.predict(X_transformed_subset)

    if predictions == [0]:
        decision = 'No' 
    else:
        decision = "Yes"
        
    """    
    plt.figure()
    plt.hist(image.ravel(), bins=256, range=(0, 250), histtype='stepfilled', density=True)
    plt.title('Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()
    """

    return decision

def calculate_glcm_properties(image):
    distances = [1]
    
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm_properties = []
    glcm_energies = []
    glcm_homogeneities = []

    for angle in angles:
        glcm = greycomatrix(image, distances=distances, angles=[angle], symmetric=True, normed=True)
        
        energy = greycoprops(glcm, 'energy')[0, 0]
        homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
        glcm_energies.append(energy)
        glcm_homogeneities.append(homogeneity)

    return glcm_energies, glcm_homogeneities


def model_2(image):
    pca_model_2 = r'C:\Users\aldif\OneDrive\Documents\Semester VII\Python\PyQt\MCC\Inspection\pca_model_2.pkl'
    classifier_model_2 = r'C:\Users\aldif\OneDrive\Documents\Semester VII\Python\PyQt\MCC\Inspection\model_2_RF.h5'
    
    pca = joblib.load(pca_model_2)
    rf_classifier = joblib.load(classifier_model_2)
    
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    X = hist[0:50]
    
    energy_values, homogeneity_values = calculate_glcm_properties(image)
    combined_data = np.hstack((energy_values, homogeneity_values, X))

    X = combined_data.reshape(1, -1)
    X_gabungan = pca.transform(X)        
    X_gabungan = X_gabungan[:, 0:4]
    
    predictions = rf_classifier.predict(X_gabungan)

    if predictions == [0]:
        decision = 'No'
    else :
        decision = "Yes"

    return decision

def model_3(image_asli):
    image = np.copy(image_asli)
    image[image == 255] = 0
    
    num_objects = 0
    
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    corner_coordinates = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x2, y2 = x + w, y + h
        
        contour_area = cv2.contourArea(contour)
        
        min_contour_area = 1
        if contour_area >= min_contour_area:
            corner_coordinates.append(((x, y), (x2, y2)))
            
    if corner_coordinates:
        image_with_boxes = cv2.cvtColor(image_asli, cv2.COLOR_GRAY2BGR)
        for i, ((x, y), (x2, y2)) in enumerate(corner_coordinates, start=1):
            cv2.rectangle(image_with_boxes, (x, y), (x2, y2), (0, 0, 255), 2)
        
        num_objects = len(corner_coordinates)
    else:
        print("No Bottle found in the image")

    return num_objects, corner_coordinates
        
        
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = individualCamera()
    window.show()
    sys.exit(app.exec_())

        