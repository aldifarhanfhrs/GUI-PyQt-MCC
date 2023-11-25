import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QCheckBox, QComboBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer

class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.camera = cv2.VideoCapture(0)
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.checkbox = QCheckBox('Aktifkan Kamera', self)
        self.checkbox.stateChanged.connect(self.toggle_camera)

        self.camera_combobox = QComboBox(self)
        self.populate_camera_combobox()
        self.camera_combobox.currentIndexChanged.connect(self.select_camera)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.checkbox)
        self.layout.addWidget(self.camera_combobox)

        self.central_widget = QWidget()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.is_camera_active = False

    def populate_camera_combobox(self):
        # Add available camera indexes to the combo box
        for i in range(10):
            if cv2.VideoCapture(i).isOpened():
                self.camera_combobox.addItem(f'Camera {i}')
                cv2.VideoCapture(i).release()

    def select_camera(self, index):
        # Set the selected camera index
        self.camera.release()
        self.camera = cv2.VideoCapture(index)

    def toggle_camera(self, state):
        self.is_camera_active = state == Qt.Checked
        if self.is_camera_active:
            self.timer.start(30)  # Update every 30 milliseconds
        else:
            self.timer.stop()

    def update_frame(self):
        ret, frame = self.camera.read()
        if ret:
            # Convert the OpenCV image to a format suitable for QPixmap
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(pixmap)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CameraApp()
    window.setGeometry(100, 100, 800, 600)
    window.setWindowTitle('Camera App with PyQt and OpenCV')
    window.show()
    sys.exit(app.exec_())
