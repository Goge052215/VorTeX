from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QFrame
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from manim import *
import cv2
import logging
import numpy as np
import tempfile
import os
from manim_visual.manim_visualizer import MathVisualizer

# Configure Manim for video output
config.media_width = "100%"
config.preview = True
config.write_to_movie = True
config.format = "mp4"
config.save_last_frame = False

class VisualizationWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_animation_frame)
        self.video_capture = None
        self.animation_speed = 30  # FPS
        self.current_func = "x**2"
        
        # Instantiate MathVisualizer
        self.visualizer = MathVisualizer()
        
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the visualization window UI."""
        self.setWindowTitle('Mathematical Visualization')
        self.setGeometry(200, 200, 800, 600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Controls layout
        controls_layout = QHBoxLayout()
        
        # X Range Input
        self.x_range_label = QLabel("X Range:")
        self.x_range_input = QLineEdit("(-10, 10)")
        controls_layout.addWidget(self.x_range_label)
        controls_layout.addWidget(self.x_range_input)
        
        # Y Range Input
        self.y_range_label = QLabel("Y Range:")
        self.y_range_input = QLineEdit("(-5, 5)")
        controls_layout.addWidget(self.y_range_label)
        controls_layout.addWidget(self.y_range_input)
        
        # Animation controls
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_animation)
        controls_layout.addWidget(self.play_button)
        
        # Update Button
        self.update_button = QPushButton("Update Plot")
        self.update_button.clicked.connect(self.update_plot)
        controls_layout.addWidget(self.update_button)
        
        main_layout.addLayout(controls_layout)
        
        # Visualization area with a border
        self.viz_frame = QFrame()
        self.viz_frame.setFrameShape(QFrame.Box)
        self.viz_frame.setLineWidth(2)
        
        self.viz_view = QGraphicsView(self.viz_frame)
        self.viz_scene = QGraphicsScene()
        self.viz_view.setScene(self.viz_scene)
        
        main_layout.addWidget(self.viz_frame)

    def toggle_animation(self):
        """Toggle animation playback."""
        if self.animation_timer.isActive():
            self.animation_timer.stop()
            self.play_button.setText("Play")
        else:
            if self.video_capture is not None:
                self.animation_timer.start(1000 // self.animation_speed)
                self.play_button.setText("Pause")

    def update_animation_frame(self):
        """Update the current frame of the animation."""
        if self.video_capture is not None and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if ret:
                self.display_frame(frame)
            else:
                # Reset to beginning of video when it ends
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def display_frame(self, frame):
        """Display a single frame in the visualization area."""
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = frame_rgb.shape[:2]
            bytes_per_line = 3 * width
            
            # Create QImage from frame
            image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            
            # Clear previous frame and display new one
            self.viz_scene.clear()
            self.viz_scene.addPixmap(pixmap)
            
            # Fit the video to the view and center it
            self.viz_view.fitInView(self.viz_scene.sceneRect(), Qt.KeepAspectRatio)
            self.viz_view.setAlignment(Qt.AlignCenter)
            
        except Exception as e:
            self.logger.error(f"Error displaying frame: {e}")

    def display_manim_scene(self, scene_class, x_range=(-10, 10), y_range=(-5, 5)):
        """Display a Manim scene in the visualization area."""
        try:
            # Create temporary directory
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Set Manim configuration for this render
                config.media_dir = tmp_dir
                config.video_dir = tmp_dir
                
                # Create and render the scene
                scene = scene_class(x_range=x_range, y_range=y_range)
                scene.render()
                
                # Get the exact path to the rendered video
                video_filename = f"{scene.__class__.__name__}.mp4"
                video_path = os.path.join(tmp_dir, "videos", video_filename)
                
                # Ensure the video file exists
                if not os.path.exists(video_path):
                    self.logger.error(f"Video file not found at {video_path}")
                    return
                
                self.logger.info(f"Video file created at: {video_path}")
                
                # Close any existing video capture
                if self.video_capture is not None:
                    self.video_capture.release()
                
                # Open the video file
                self.video_capture = cv2.VideoCapture(video_path)
                
                if not self.video_capture.isOpened():
                    self.logger.error("Failed to open video capture")
                    return
                
                # Read and display the first frame
                ret, frame = self.video_capture.read()
                if ret:
                    self.display_frame(frame)
                    # Start animation
                    self.animation_timer.start(1000 // self.animation_speed)
                    self.play_button.setText("Pause")
                else:
                    self.logger.error("Failed to read first frame")
                
        except Exception as e:
            self.logger.error(f"Error displaying Manim scene: {e}")
            raise  # Re-raise the exception for debugging

    def update_plot(self):
        """Update the plot based on user input."""
        try:
            x_range = eval(self.x_range_input.text())
            y_range = eval(self.y_range_input.text())
            # Use MathVisualizer to visualize the function
            self.visualizer.visualize_function(self.current_func, x_range, y_range)
        except Exception as e:
            self.logger.error(f"Error updating plot: {e}")

    def visualize_function(self, func_str: str, x_range: tuple = (-10, 10), y_range: tuple = (-5, 5)):
        """Visualize a function using Manim."""
        self.current_func = func_str
        
        class FunctionScene(Scene):
            def __init__(self, x_range, y_range, **kwargs):
                super().__init__(**kwargs)
                self.x_range = x_range
                self.y_range = y_range

            def construct(self):
                axes = Axes(
                    x_range=self.x_range,
                    y_range=self.y_range,
                    axis_config={"color": BLUE},
                    x_length=10,
                    y_length=6
                )
                
                try:
                    # Log the function string for debugging
                    self.logger.debug(f"Evaluating function: {func_str}")
                    
                    # Use a safe eval context
                    safe_dict = {"x": 0, "np": np}
                    graph = axes.plot(lambda x: eval(func_str, {"x": x, "np": np}, safe_dict))
                    
                    # Create animation
                    self.play(Create(axes))
                    self.play(Create(graph))
                except Exception as e:
                    self.add(Text(f"Error: {str(e)}"))
                    self.logger.error(f"Error in function visualization: {e}")
        
        self.display_manim_scene(FunctionScene, x_range, y_range)

    def resizeEvent(self, event):
        """Handle window resize events."""
        super().resizeEvent(event)
        if hasattr(self, 'viz_view'):
            self.viz_view.fitInView(self.viz_scene.sceneRect(), Qt.KeepAspectRatio)

    def closeEvent(self, event):
        """Clean up resources when closing the window."""
        try:
            # Stop animation timer
            self.animation_timer.stop()
            
            # Release video capture
            if self.video_capture is not None:
                self.video_capture.release()
                self.video_capture = None
                
            # Clean up Manim files
            self._cleanup_manim_files()
            
            super().closeEvent(event)
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def _cleanup_manim_files(self):
        """Clean up all Manim-generated files."""
        try:
            # Get the temporary directory path
            temp_dir = config.media_dir
            
            # List of file extensions to clean up
            extensions = ['.mp4', '.svg', '.tex', '.png', '.jpg', '.aux', '.log', '.dvi']
            
            # Walk through all subdirectories
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if any(file.endswith(ext) for ext in extensions):
                        try:
                            file_path = os.path.join(root, file)
                            os.remove(file_path)
                            self.logger.debug(f"Removed file: {file_path}")
                        except Exception as e:
                            self.logger.warning(f"Failed to remove file {file}: {e}")
                            
            self.logger.info("Cleaned up Manim files")
        except Exception as e:
            self.logger.error(f"Error cleaning up Manim files: {e}")
