import os
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QAction, QFileDialog,
                             QVBoxLayout, QLabel, QWidget, QScrollArea,
                             QHBoxLayout, QMenu, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np


class FaceExtractor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Passport Photo Extractor')
        self.setGeometry(100, 100, 800, 600)

        self.selected_model = 'Haar Cascade'  # Default model selection
        self.net = self.load_haar_model()  # Load default model
        self.initUI()

    def initUI(self):
        # Create central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Create a QScrollArea for face images
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area_content = QWidget()
        self.scroll_area_layout = QVBoxLayout(self.scroll_area_content)  # Vertical layout for wrapping rows
        self.scroll_area.setWidget(self.scroll_area_content)
        self.layout.addWidget(self.scroll_area)

        # Initialize the image row layout
        self.image_row_layout = QHBoxLayout()  # This will hold one row of images
        self.scroll_area_layout.addLayout(self.image_row_layout)  # Add the row layout to the scroll area layout

        # Create menu actions
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')

        # New action to clear the table view
        new_action = QAction('New', self)
        new_action.triggered.connect(self.clear_images)
        file_menu.addAction(new_action)

        # Open action
        open_action = QAction('Open', self)  # Changed to 'Open'
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)

        # Exit action
        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)  # Close the application
        file_menu.addAction(exit_action)

        # Options menu
        options_menu = menubar.addMenu('Options')
        haar_action = QAction('Haar Cascade', self, checkable=True)
        haar_action.setChecked(True)  # Default is Haar
        haar_action.triggered.connect(self.select_model)  # Direct connection
        options_menu.addAction(haar_action)

        dnn_action = QAction('DNN Model', self, checkable=True)
        dnn_action.triggered.connect(self.select_model)  # Direct connection
        options_menu.addAction(dnn_action)

        # Toolbar button
        toolbar = self.addToolBar('Toolbar')
        toolbar.addAction(open_action)

        # Set drag and drop
        self.setAcceptDrops(True)

    def clear_images(self):
        # Clear all images in the display
        for i in reversed(range(self.image_row_layout.count())):
            widget = self.image_row_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        # Reset the image row layout
        self.image_row_layout = QHBoxLayout()  # Reset to a new row layout
        self.scroll_area_layout.addLayout(self.image_row_layout)  # Add new row to the scroll area layout

    def load_haar_model(self):
        haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(haar_cascade_path):
            raise FileNotFoundError(f"Haar cascade file not found: {haar_cascade_path}")
        return cv2.CascadeClassifier(haar_cascade_path)

    def load_dnn_model(self):
        net = cv2.dnn.readNetFromCaffe(r'E:\Development\Projects\FaceFinder\models\deploy.prototxt',
                                       r'E:\Development\Projects\FaceFinder\models\res10_300x300_ssd_iter_140000.caffemodel')
        print("DNN model loaded successfully.")
        return net

    def select_model(self):
        action = self.sender()  # Get the action that triggered this method
        model = action.text()  # Get the text of the action
        self.selected_model = model  # Update selected model based on the action text

        if model == 'Haar Cascade':
            self.net = self.load_haar_model()
        else:  # DNN Model
            self.net = self.load_dnn_model()

        # Update the menu check states based on selected_model
        for action in self.menuBar().findChildren(QAction):
            if action.text() in ['Haar Cascade', 'DNN Model']:
                action.setChecked(action.text() == self.selected_model)  # Set checked state based on current selection

        print(f"Selected model: {self.selected_model}")  # Print the selected model

    def open_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Images (*.png *.xpm *.jpg *.jpeg)')
        if file_name:
            self.process_image(file_name)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        # Clear previous faces from the current image row layout
        self.clear_images()  # Clear existing images before adding new ones

        for url in event.mimeData().urls():
            self.process_image(url.toLocalFile())

    def detect_faces_haar(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.net.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
        print(f"Detected faces using Haar: {len(faces)}")
        return faces  # Return in (x, y, w, h) format

    def detect_faces_dnn(self, image):
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        scores = []  # List to hold confidence scores
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Ensure coordinates are within image bounds
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)

                # Check if bounding box is valid
                if endX > startX and endY > startY:
                    faces.append((startX, startY, endX, endY))
                    scores.append(float(confidence))  # Store the confidence score

        # Apply Non-Maximum Suppression
        if len(faces) > 0:
            indices = cv2.dnn.NMSBoxes(faces, scores, score_threshold=0.5, nms_threshold=0.4)

            # Check if indices is a valid array before processing
            if len(indices) > 0:
                # Correctly extract faces based on indices
                faces = [faces[i] for i in indices.flatten().tolist()]  # Use flatten to get the correct indices
            else:
                print("No valid indices returned from NMS.")

        print(f"Detected faces using DNN: {len(faces)}")
        return faces  # Return in (startX, startY, endX, endY) format

    def detect_faces(self, image):
        if self.selected_model == 'Haar Cascade':
            return self.detect_faces_haar(image)
        else:  # DNN Model
            return self.detect_faces_dnn(image)

    def process_image_haar(self, file_name):
        image = cv2.imread(file_name)

        if image is None:
            QMessageBox.critical(self, "Error", "Failed to load image. Please check the file format and try again.")
            return

        faces = self.detect_faces_haar(image)  # Call Haar detection

        # Clear previous faces
        self.clear_images()  # Clear existing images before adding new ones

        for index, (x, y, w, h) in enumerate(faces):
            # Extract the face image using correct coordinates
            face_image = image[y:y + h, x:x + w]

            # Check if face_image is valid
            if face_image.size > 0:  # Ensure we have a valid image
                print(f"Face Image {index}: Start ({x}, {y}), Size: {face_image.shape}")  # Debugging output
                if self.is_good_quality(face_image):  # Implement this method to check quality
                    self.display_face(face_image, index)  # Pass the index as an argument
            else:
                print(f"Skipping empty face image at coordinates: ({x}, {y}, {w}, {h}).")

    def process_image_dnn(self, file_name):
        image = cv2.imread(file_name)

        if image is None:
            QMessageBox.critical(self, "Error", "Failed to load image. Please check the file format and try again.")
            return

        faces = self.detect_faces_dnn(image)  # Call DNN detection

        # Clear previous faces
        self.clear_images()  # Clear existing images before adding new ones

        for index, (startX, startY, endX, endY) in enumerate(faces):
            # Ensure valid coordinates
            if startX >= 0 and startY >= 0 and endX > startX and endY > startY:
                # Extract the face image using correct coordinates
                face_image = image[startY:endY, startX:endX]

                # Check if face_image is valid
                if face_image.size > 0:  # Ensure we have a valid image
                    print(
                        f"Face Image {index}: Start ({startX}, {startY}), Size: {face_image.shape}")  # Debugging output
                    if self.is_good_quality(face_image):  # Implement this method to check quality
                        self.display_face(face_image, index)  # Pass the index as an argument
                else:
                    print(
                        f"Skipping empty face image at coordinates: ({startX}, {startY}, {endX - startX}, {endY - startY}).")
            else:
                print(f"Skipping invalid bounding box for face image: ({startX}, {startY}, {endX}, {endY}).")

    def process_image(self, file_name):
        if self.selected_model == 'Haar Cascade':
            self.process_image_haar(file_name)  # Use Haar processing
        else:  # DNN Model
            self.process_image_dnn(file_name)  # Use DNN processing

    def display_face(self, face_image, index):
        if face_image.size == 0:
            print(f"Skipping empty face image at index {index}.")
            return  # Skip if the face image is empty

        face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        height, width, channel = face_image_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(face_image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        label = QLabel()
        label.setPixmap(pixmap.scaled(100, 100, Qt.KeepAspectRatio))  # Adjust size as needed

        # Add label to the current row layout
        self.image_row_layout.addWidget(label)

        # Check if we need to wrap to a new row
        if (index + 1) % 5 == 0:  # Adjust the number of images per row as needed
            self.image_row_layout = QHBoxLayout()  # Create a new row layout
            self.scroll_area_layout.addLayout(self.image_row_layout)  # Add new row to the scroll area layout

    def show_context_menu(self, pos, face_image):
        context_menu = QMenu(self)
        save_action = QAction('Save As', self)
        save_action.triggered.connect(lambda: self.save_face(face_image))
        context_menu.addAction(save_action)
        context_menu.exec_(self.scroll_area.viewport().mapToGlobal(pos))

    def save_face(self, face_image):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Image", "",
                                                   "JPEG Files (*.jpg *.jpeg);;PNG Files (*.png)", options=options)
        if file_name:
            cv2.imwrite(file_name, face_image)

    def is_good_quality(self, face_image):
        # Placeholder for quality check
        return True  # Replace with actual logic


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceExtractor()
    window.show()
    sys.exit(app.exec_())
