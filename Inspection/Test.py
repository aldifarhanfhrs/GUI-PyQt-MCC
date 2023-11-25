import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Window dengan Logo")

        # Membuat QLabel untuk menampilkan logo
        logo_label = QLabel(self)
        
        # Menentukan path logo (ganti dengan path logo yang sesuai)
        logo_path = "C:\Users\aldif\OneDrive\Documents\Semester VII\Python\PyQt\MCC\Individual Inspection\logo MCC.png"

        # Membuat objek QPixmap dari path logo
        logo_pixmap = QPixmap(logo_path)

        # Menampilkan gambar pada QLabel
        logo_label.setPixmap(logo_pixmap)
        logo_label.setAlignment(Qt.AlignCenter)

        # Membuat layout utama
        main_layout = QVBoxLayout()
        main_layout.addWidget(logo_label)

        # Membuat widget utama dan menetapkan layout utama
        central_widget = QWidget()
        central_widget.setLayout(main_layout)

        # Menetapkan widget utama ke window
        self.setCentralWidget(central_widget)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
