import sys
import vispy
from vispy import app, scene
from vispy.util import keys
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QHBoxLayout

class VispyWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Vispy Canvas
        self.canvas = scene.SceneCanvas(keys='interactive', show=True)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.TurntableCamera(fov=60.0, elevation=30.0, azimuth=30.0)
        # self.grid = scene.Grid(size=10, parent=self.view)

        # Example Vispy content (a cube)
        self.cube = scene.visuals.Box(color=(0.5, 0.5, 1, 1), parent=self.view)

        # Connect Vispy events (example: key press)
        self.canvas.events.key_press.connect(self.on_key_press)

        # PyQt Layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas.native)  # Embed Vispy canvas

    def on_key_press(self, event):
        if event.key == keys.SPACE:
            self.cube.shared_program.frag['a_color'] = (1, 0, 0, 1) # Example change


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vispy in PyQt6")

        # Main layout (horizontal)
        main_layout = QHBoxLayout()

        # Left panel (PyQt widgets)
        left_panel = QWidget()
        left_layout = QVBoxLayout()

        self.foo_button = QPushButton("Foo")
        self.foo_button.clicked.connect(self.on_foo_clicked)
        left_layout.addWidget(self.foo_button)

        left_panel.setLayout(left_layout)

        # Right panel (Vispy canvas)
        self.vispy_widget = VispyWidget()

        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.vispy_widget)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def on_foo_clicked(self):
        print("Foo button clicked!")
        self.vispy_widget.cube.shared_program.frag['a_color'] = (0, 1, 0, 1)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()

    # Integrate Vispy app with PyQt event loop
    sys.exit(app.exec())