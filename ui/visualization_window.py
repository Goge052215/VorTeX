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
import re
import tempfile
import atexit
import uuid

class VisualizationWindow(QMainWindow):
    # Track all created media directories for cleanup
    _media_dirs = []
    
    @classmethod
    def cleanup_all_media_dirs(cls):
        """Clean up all media directories when the program exits"""
        for media_dir in cls._media_dirs:
            try:
                if os.path.exists(media_dir):
                    shutil.rmtree(media_dir, ignore_errors=True)
                    logging.getLogger(__name__).debug(f"Cleaned up media directory: {media_dir}")
            except Exception as e:
                logging.getLogger(__name__).error(f"Error cleaning up media directory {media_dir}: {e}")
        
        # Clear the list
        cls._media_dirs.clear()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.current_function = None
        self.current_python_expr = None
        
        # Register cleanup on exit if not already registered
        if not hasattr(VisualizationWindow, "_cleanup_registered"):
            atexit.register(VisualizationWindow.cleanup_all_media_dirs)
            VisualizationWindow._cleanup_registered = True
            
        # Create a unique media directory for this instance
        self.media_dir = os.path.join(tempfile.gettempdir(), f"vortex_vis_{uuid.uuid4().hex}")
        VisualizationWindow._media_dirs.append(self.media_dir)
        
        # Media player and video widget
        self.media_player = QMediaPlayer(self)
        self.video_widget = QVideoWidget(self)
        self.media_player.setVideoOutput(self.video_widget)
        
        self.media_player.stateChanged.connect(self.media_state_changed)
        self.media_player.positionChanged.connect(self.position_changed)
        self.media_player.durationChanged.connect(self.duration_changed)
        
        self.visualizer = MathVisualizer()
        
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the visualization window UI."""
        self.setWindowTitle('Mathematical Visualization')
        self.setGeometry(200, 200, 1000, 600)
        
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
            if os.path.exists(self.media_dir):
                # Only remove contents, not the directory itself
                for filename in os.listdir(self.media_dir):
                    file_path = os.path.join(self.media_dir, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        self.logger.error(f"Error removing {file_path}: {e}")
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
                
                # Get the current expression from the function input
                updated_func = self.function_input.text()
                
                # Format it for SymPy if it has been edited
                if updated_func != self.current_function:
                    self.current_function = updated_func
                    self.current_python_expr = self._format_expression_for_sympy(updated_func)
                
                self.visualize_function(self.current_function, self.current_python_expr, x_range, y_range)
            else:
                QMessageBox.warning(self, "Error", "No function to visualize.")
        except Exception as e:
            self.logger.error(f"Error updating plot: {e}")
            QMessageBox.critical(self, "Error", f"Failed to update plot: {str(e)}")

    def visualize_function(self, func_str: str, python_expr: str = None, x_range: tuple = (-10, 10), y_range: tuple = (-5, 5)):
        """Visualize a function using MathVisualizer."""
        try:
            if not func_str:
                raise ValueError("No expression provided")
                
            self.logger.debug(f"Visualizing function: {func_str}")
            
            self.media_player.stop()
            self.media_player.setMedia(QMediaContent())
            
            self._cleanup_manim_files()
            
            # If no Python expression is provided, format the expression for SymPy
            if python_expr is None:
                python_expr = self._format_expression_for_sympy(func_str)
                
            self.logger.debug(f"Formatted expression for visualization: {python_expr}")
            
            # Set current function values
            self.current_function = func_str
            self.current_python_expr = python_expr
            self.function_input.setText(func_str)
            self.x_range_input.setText(str(x_range))
            self.y_range_input.setText(str(y_range))
            
            # Ensure the media directory exists
            video_dir = os.path.join(self.media_dir, "videos", "1080p60")
            os.makedirs(video_dir, exist_ok=True)
            
            # Configure manim to use our media directory
            config.media_dir = self.media_dir
            config.video_dir = video_dir
            
            scene = self.visualizer.FunctionScene(
                self.current_python_expr,
                x_range=x_range,
                y_range=y_range, 
                display_text=self.current_function,
                logger=self.logger
            )
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
            self.logger.error(f"Error visualizing function: {e}")
            QMessageBox.critical(self, "Error", f"Failed to visualize function: {str(e)}")

    def _handle_media_status(self, status):
        """Handle media status changes."""
        if status == QMediaPlayer.EndOfMedia:
            duration = self.media_player.duration()
            if duration > 0:
                self.media_player.setPosition(duration - 1)
                self.media_player.pause()
            self.play_button.setText("Play")

    def set_expression(self, expression):
        """Set the expression to visualize"""
        try:
            if expression:
                # Set the function input text
                self.function_input.setText(expression)
                
                # Format the expression for SymPy (add explicit multiplication)
                python_expr = self._format_expression_for_sympy(expression)
                
                # Visualize the function
                self.visualize_function(expression, python_expr)
        except Exception as e:
            self.logger.error(f"Error setting expression: {e}")
            QMessageBox.critical(self, "Error", f"Failed to set expression: {str(e)}")
            
    def _format_expression_for_sympy(self, expr):
        """Format a mathematical expression for SymPy parsing."""
        if not expr or expr.strip() == "":
            return "x"
        
        formatted_expr = expr.strip()
        self.logger.debug(f"Original expression: '{expr}'")
        
        # Handle power operations (replace x^2 with x**2)
        formatted_expr = re.sub(r'([a-zA-Z0-9_\)]+)\s*\^\s*([0-9a-zA-Z_\(]+)', r'\1**\2', formatted_expr)
        
        # Add explicit multiplication between numbers and variables
        formatted_expr = re.sub(r'(\d+)([a-zA-Z\(])', r'\1*\2', formatted_expr)
        
        # Handle special cases like (2)(x) or (x)(y)
        formatted_expr = re.sub(r'\)(\()', r')*(', formatted_expr)
        
        # Handle implicit multiplication between variables
        # But be careful not to break function names like 'sin'
        def add_multiplication(match):
            a, b = match.groups()
            # Check if this might be part of a function name
            common_funcs = ['sin', 'cos', 'tan', 'exp', 'log', 'ln', 'sqrt', 
                           'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh']
            
            # Don't add multiplication if it would break a function name
            for func in common_funcs:
                if (a + b) in func:
                    return a + b
            
            return f"{a}*{b}"
            
        formatted_expr = re.sub(r'([a-zA-Z])([a-zA-Z])', add_multiplication, formatted_expr)
        
        # Fix the implicit multiplication between variables and function calls
        # This regex looks for patterns like 'xsin(' and converts to 'x*sin('
        for func in ['sin', 'cos', 'tan', 'exp', 'log', 'ln', 'sqrt', 'asin', 'acos', 'atan']:
            formatted_expr = re.sub(rf'([a-zA-Z0-9_\)])({func})\(', r'\1*\2(', formatted_expr)
        
        # Map function names to SymPy equivalents
        replacements = {
            'ln': 'log',  # ln is log in SymPy
            'pi': 'pi',
            'e': 'E'
        }
        
        # Apply replacements of common constants
        for old, new in replacements.items():
            # Only replace when it's a whole word (not part of another word)
            formatted_expr = re.sub(rf'\b{old}\b', new, formatted_expr)
        
        # Do a basic validation check
        if formatted_expr.startswith('**'):
            # Invalid syntax (power operator at start)
            formatted_expr = formatted_expr.lstrip('*')
        
        self.logger.debug(f"Formatted expression for SymPy: '{formatted_expr}'")
        return formatted_expr

    def closeEvent(self, event):
        """Handle window close event."""
        try:
            self.media_player.stop()
            self.media_player.setMedia(QMediaContent())
            
            # Clean up all files created by this instance
            self._cleanup_manim_files()
            
            event.accept()
            
            if hasattr(self.parent(), 'viz_window'):
                self.parent().viz_window = None
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
