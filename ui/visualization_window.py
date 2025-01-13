from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QLineEdit, QPushButton, QMessageBox, QSlider
)
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from manim_visual.manim_visualizer import MathVisualizer
from manim import *
import logging
import os
import shutil

class VisualizationWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.current_function = None
        
        # Media player and video widget
        self.media_player = QMediaPlayer(self)
        self.video_widget = QVideoWidget(self)
        self.media_player.setVideoOutput(self.video_widget)
        
        self.media_player.stateChanged.connect(self.media_state_changed)
        self.media_player.positionChanged.connect(self.position_changed)
        self.media_player.durationChanged.connect(self.duration_changed)
        
        self.visualizer = MathVisualizer()
        
        self._init_ui()
        
        self._cleanup_manim_files()

    def _init_ui(self):
        """Initialize the visualization window UI."""
        self.setWindowTitle('Mathematical Visualization')
        self.setGeometry(200, 200, 800, 600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        controls_layout = QHBoxLayout()
        
        self.x_range_label = QLabel("X Range:")
        self.x_range_input = QLineEdit("(-10, 10)")
        controls_layout.addWidget(self.x_range_label)
        controls_layout.addWidget(self.x_range_input)
        
        self.y_range_label = QLabel("Y Range:")
        self.y_range_input = QLineEdit("(-5, 5)")
        controls_layout.addWidget(self.y_range_label)
        controls_layout.addWidget(self.y_range_input)
        
        # Animation controls
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_animation)
        controls_layout.addWidget(self.play_button)
        
        self.replay_button = QPushButton("Replay")
        self.replay_button.clicked.connect(self.replay_animation)
        controls_layout.addWidget(self.replay_button)
        
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self.set_position)
        controls_layout.addWidget(self.position_slider)
        
        self.update_button = QPushButton("Update Plot")
        self.update_button.clicked.connect(self.update_plot)
        controls_layout.addWidget(self.update_button)
        
        self.function_label = QLabel("Function:")
        self.function_input = QLineEdit()
        controls_layout.addWidget(self.function_label)
        controls_layout.addWidget(self.function_input)
        
        main_layout.addLayout(controls_layout)
        main_layout.addWidget(self.video_widget)

    def _cleanup_manim_files(self):
        """Clean up all files in the media directory."""
        try:
            media_dir = os.path.join(os.getcwd(), "media")
            if os.path.exists(media_dir):
                shutil.rmtree(media_dir, ignore_errors=True)
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def media_state_changed(self, state):
        """Handle media player state changes."""
        if state == QMediaPlayer.PlayingState:
            self.play_button.setText("Pause")
        else:
            self.play_button.setText("Play")
            
        if state == QMediaPlayer.StoppedState:
            duration = self.media_player.duration()
            if duration > 0:
                self.media_player.setPosition(duration - 1)
                self.media_player.pause()
            self.play_button.setText("Play")

    def position_changed(self, position):
        """Update slider position."""
        self.position_slider.setValue(position)

    def duration_changed(self, duration):
        """Update slider range."""
        self.position_slider.setRange(0, duration)

    def set_position(self, position):
        """Set media player position."""
        self.media_player.setPosition(position)

    def toggle_animation(self):
        """Toggle between play and pause."""
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

    def replay_animation(self):
        """Replay the animation from the beginning."""
        self.media_player.setPosition(0)
        self.media_player.play()
        self.play_button.setText("Pause")

    def update_plot(self):
        """Update the plot based on user input."""
        try:
            if self.current_function:
                self.media_player.stop()
                self.media_player.setMedia(QMediaContent())
                self._cleanup_manim_files()
                
                x_range = eval(self.x_range_input.text())
                y_range = eval(self.y_range_input.text())
                self.visualize_function(self.current_function, x_range, y_range)
            else:
                QMessageBox.warning(self, "Error", "No function to visualize.")
        except Exception as e:
            self.logger.error(f"Error updating plot: {e}")
            QMessageBox.critical(self, "Error", f"Failed to update plot: {str(e)}")

    def visualize_function(self, func_str: str, x_range: tuple = (-10, 10), y_range: tuple = (-5, 5)):
        """Visualize a function using MathVisualizer."""
        try:
            self.media_player.stop()
            self.media_player.setMedia(QMediaContent())
            
            self._cleanup_manim_files()
            
            self.current_function = func_str
            self.function_input.setText(func_str)
            self.x_range_input.setText(str(x_range))
            self.y_range_input.setText(str(y_range))
            
            media_dir = os.path.join(os.getcwd(), "media")
            video_dir = os.path.join(media_dir, "videos", "1080p60")
            os.makedirs(video_dir, exist_ok=True)
            
            config.media_dir = media_dir
            config.video_dir = video_dir
            
            scene = self.visualizer.FunctionScene(func_str, x_range=x_range, y_range=y_range, logger=self.logger)
            scene.render()
            
            video_path = os.path.join(video_dir, "FunctionScene.mp4")
            
            if os.path.exists(video_path):
                self.media_player.setMedia(
                    QMediaContent(QUrl.fromLocalFile(os.path.abspath(video_path)))
                )
                self.play_button.setEnabled(True)
                self.replay_button.setEnabled(True)
                
                self.media_player.mediaStatusChanged.connect(self._handle_media_status)
                
                self.media_player.play()
                self.play_button.setText("Pause")
            else:
                self.logger.error(f"Video file not found at {video_path}")
                QMessageBox.critical(self, "Error", "Video file not found")
                
        except Exception as e:
            self.logger.error(f"Error in visualization: {e}")
            QMessageBox.critical(self, "Error", f"Failed to visualize function: {str(e)}")

    def _handle_media_status(self, status):
        """Handle media status changes."""
        if status == QMediaPlayer.EndOfMedia:
            duration = self.media_player.duration()
            if duration > 0:
                self.media_player.setPosition(duration - 1)
                self.media_player.pause()
            self.play_button.setText("Play")

    def closeEvent(self, event):
        """Handle window close event."""
        try:
            self.media_player.stop()
            self.media_player.setMedia(QMediaContent())
            
            self._cleanup_manim_files()
            
            event.accept()
            
            if hasattr(self.parent(), 'viz_window'):
                self.parent().viz_window = None
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
