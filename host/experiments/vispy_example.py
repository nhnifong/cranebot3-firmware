import sys

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt6.QtCore import QTimer
import vispy.scene
from vispy.scene import visuals
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Vispy in PyQt6")

        # Create a central widget to hold the Vispy canvas
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create a layout for the central widget
        layout = QVBoxLayout(central_widget)

        # Create a Vispy canvas
        canvas = vispy.scene.SceneCanvas(keys='interactive')
        layout.addWidget(canvas.native)

        # Create a viewbox
        view = canvas.central_widget.add_view()

        # Create a visual (e.g., a scatter plot)
        self.scatter = visuals.Markers()
        view.add(self.scatter)

        # Generate some random data
        n = 100
        pos = np.random.normal(size=(n, 2), scale=0.2)
        self.scatter.set_data(pos, face_color=(1, 1, 1, 0.5), size=10)

        # Update the plot periodically
        timer = QTimer(self)
        timer.timeout.connect(self.update_plot)
        timer.start(50)

    def update_plot(self):
        # Update the data for the scatter plot (e.g., with new random values)
        n = 100
        pos = np.random.normal(size=(n, 2), scale=200)
        self.scatter.set_data(pos, face_color=(1, 1, 1, 0.5), size=10)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())