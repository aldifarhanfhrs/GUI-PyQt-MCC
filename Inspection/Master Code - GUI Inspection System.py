__author__      = "Muhammad Masdar Mahasin"
__copyright__   = "Copyright Brawijaya University & Philips @2023"
__version__ = "1.0.0"

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
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QApplication

from PyQt5.uic import loadUi
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
import pyqtgraph as pg

import io
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def model_1(image):
    pca_model_1 = r'C:\Users\aldif\OneDrive\Documents\Semester VII\Python\PyQt\MCC\Inspection\pca_model_1.pkl'
    classifier_model_1 = r'C:\Users\aldif\OneDrive\Documents\Semester VII\Python\PyQt\MCC\Inspection\model_1_RF.h5'
    
    pca = joblib.load(pca_model_1)
    rf_classifier = joblib.load(classifier_model_1)
    
    hist, bins = np.histogram(image, bins=256, range=(0, 256))
    X = hist[20:100]
    X = X.reshape(1, -1)
    
    X_transformed = pca.transform(X)
    
    X_transformed_subset = X_transformed[:, 0:3]
    predictions = rf_classifier.predict(X_transformed_subset)

    if predictions == [0]:
        kesimpulan = 'No' 
    else:
        kesimpulan = "Yes"

    return kesimpulan

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
        kesimpulan = 'No'
    else :
        kesimpulan = "Yes"

    return kesimpulan

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


class PyQtMainEntry(QMainWindow):
    def __init__(self):
        global crop_coordinates
        super().__init__()

        loadUi(r"C:\Users\aldif\OneDrive\Documents\Semester VII\Python\PyQt\MCC\Inspection\GUI_inspection_DL_system.ui", self)

        self.available_cameras = QCameraInfo.availableCameras()
        if not self.available_cameras:
            sys.exit()
        
        #crop_coordinates = (1200, 1200, 2020, 1750)

        self.comboBox.addItems(['Choose Model',
                                'Ensemble Model 1.2.0', 
                                'Future Model',])

        output = self.comboBox.currentText()

        self.comboBox_2.addItems(['Choose Camera',
                                'Choose Manual',
                                'Camera Internal',
                                'Camera External 1',
                                'Camera External 2',
                                'Camera External 3'])

        self.comboBox_2.currentIndexChanged.connect(self.select_camera)
        #self.comboBox.currentIndexChanged.connect(self.olah_citra)

        self.tombolFoto.clicked.connect(self.captureAndDisplayImage)
        self.tombolFoto.clicked.connect(self.olah_citra)

        self.tombolDirektori.clicked.connect(self.input_manual)
        self.tombolDirektori.clicked.connect(self.olah_citra)
    

    def input_manual(self):
        global fname, pixmap, img, crop_coordinates
        fname = QFileDialog.getOpenFileName(self, 'Open File', r'F:\Project\Philip Vision Checking\Data Botol\reject')
        img = Image.open(fname[0])

        crop_coordinates = (1200, 1200, 2020, 1750)
        x, y, x2, y2 = crop_coordinates

        draw = ImageDraw.Draw(img)
        draw.rectangle([(x, y), (x2, y2)], outline=(255, 0, 0), width=2)

        pixmap_img = QPixmap.fromImage(QImage(img.tobytes(), img.width, img.height, img.width * 3, QImage.Format_RGB888))

        #Open image
        img = Image.open(fname[0])

        self.labelCamera.setPixmap(pixmap_img)
        self.labelCamera.setScaledContents(True)

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
            kesimpulan = "Rejected"
        else:
            kesimpulan = "Good"

        self.label_objek.setText(str(num_objects))

        self.label_pudar.setText(hasil1)
        self.label_kurang.setText(hasil2)
        self.label_over.setText(result3_1)
        self.output_akhir.setText(kesimpulan)


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
        self.labelExtraction_2.setPixmap(pixmap_histogram)
        self.labelExtraction_2.setScaledContents(True)


        pixmap_img = QPixmap.fromImage(QImage(image.tobytes(), image.width, image.height, image.width * 3, QImage.Format_RGB888))
        #pixmap_thresholded = QPixmap.fromImage(QImage(thresholded_image.data, thresholded_image.shape[1], thresholded_image.shape[0], thresholded_image.shape[1], QImage.Format_Grayscale8))
        self.labelHistogram_2.setPixmap(pixmap_img)
        self.labelHistogram_2.setScaledContents(True)

        waktu = time.time() - start_time
        waktu = "{:.2f}".format(waktu)
        self.label_waktu.setText(str(waktu))


    def select_camera(self):
        output2 = self.comboBox_2.currentText()
        self._timer2 = QtCore.QTimer(self)
        self._timer2.timeout.connect(self._queryFrame)
        self._timer2.setInterval(10)

        if output2 == 'Camera Internal':
            self.camera = cv.VideoCapture(0)
            self.is_camera_opened = False
            self._timer2.start()

        elif output2 == 'Camera External 1':
            self.camera = cv.VideoCapture(1)
            self.is_camera_opened = False
            self._timer2.start()

        elif output2 == 'Camera External 2':
            self.camera = cv.VideoCapture(2)
            self.is_camera_opened = False
            self._timer2.start()

        elif output2 == 'Camera External 3':
            self.camera = cv.VideoCapture(3)
            self.is_camera_opened = False
            self._timer2.start()

        elif output2 == 'Choose Manual':
            # Stop the camera capture and take a picture
            self._timer2.stop()

    def captureAndDisplayImage(self):
        global image_filename, img, crop_coordinates
        if self.camera is not None:
            ret, captured_image = self.camera.read()

            if ret:
                height, width, channel = captured_image.shape
                bytesPerLine = 3 * width

                captured_image_rgb = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)

                image_live = QImage(captured_image_rgb.data, width, height, bytesPerLine, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(image_live)
                self.labelHistogram_2.setPixmap(pixmap)
                self.labelHistogram_2.setScaledContents(True)

                image_filename = r"C:\Users\aldif\OneDrive\Pictures\Camera Roll\WIN_20231112_18_23_35_Pro.jpg"
                cv2.imwrite(image_filename, captured_image)

                img = Image.open(image_filename)
                crop_coordinates = (240, 200, 400, 280)

            else:
                self.statusBar().showMessage("Failed to capture an image.")


    @QtCore.pyqtSlot()
    def _queryFrame(self):
        ret, self.frame = self.camera.read()

        img_rows, img_cols, channels = self.frame.shape
        bytesPerLine = channels * img_cols

        cv.cvtColor(self.frame, cv.COLOR_BGR2RGB, self.frame)
        QImg = QImage(self.frame.data, img_cols, img_rows, bytesPerLine, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(QImg)
        painter = QPainter(pixmap)

        #640 x 480 
        crop_coordinates = (240, 200, 400, 280)
        x, y, x2, y2 = crop_coordinates
        outline_color = (255, 0, 0)  # Red color
        line_width = 2

        pil_img = pixmap.toImage()
        pil_img = pil_img.convertToFormat(QImage.Format_RGB888)

        img = Image.fromqimage(pil_img)
        draw = ImageDraw.Draw(img)

        draw.rectangle([(x, y), (x2, y2)], outline=outline_color, width=line_width)
        modified_qimage = ImageQt.ImageQt(img)

        self.labelCamera.setPixmap(QPixmap.fromImage(modified_qimage).scaled(
            self.labelCamera.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        painter.end()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = PyQtMainEntry()
    window.show()
    sys.exit(app.exec_())