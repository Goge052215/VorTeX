"""
Legend Window module for displaying LaTeX and Matrix commands.
Provides a non-resizable window with command references.
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from ui.legend_text import SIMPLE_LATEX_LEGEND, MATRIX_LEGEND
from themes.theme_manager import ThemeManager


class LegendWindow(QWidget):
    """
    A window displaying LaTeX and Matrix command legends with themeable colors.
    
    Attributes:
        theme_manager (ThemeManager): Manages the window's theme
        theme (dict): Current theme settings
        text_color (str): Current text color
        border_color (str): Current border color
        background_color (str): Current background color
    """

    WINDOW_WIDTH = 600
    WINDOW_HEIGHT = 550
    FONT_SIZE = 13
    FONT_FAMILY = "Arial"

    def __init__(self, theme_name='aura'):
        """
        Initialize the Legend Window.

        Args:
            theme_name (str): Name of the theme to apply (default: 'aura')
        """
        super().__init__()
        self._setup_theme(theme_name)
        self._setup_window()
        self.init_ui()

    def _setup_theme(self, theme_name):
        """Initialize theme-related attributes."""
        self.theme_manager = ThemeManager()
        self.theme = self.theme_manager.apply_theme(self, theme_name)
        
        # Extract colors from theme
        self.text_color = self.theme["colors"]["text"]
        self.border_color = self.theme["colors"]["border"]
        self.background_color = self.theme["colors"]["background"]

    def _setup_window(self):
        """Configure window properties."""
        window_flags = (Qt.Window | Qt.WindowTitleHint | 
                       Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint)
        self.setWindowFlags(window_flags)
        self.setFixedSize(self.WINDOW_WIDTH, self.WINDOW_HEIGHT)

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle('Command Legend')
        self.setGeometry(200, 200, self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
        self.setStyleSheet(f"background-color: {self.background_color};")
        
        self._create_layout()

    def _create_layout(self):
        """Create and setup the window layout."""
        layout = QVBoxLayout()
        legends_layout = QHBoxLayout()

        # Create and configure legend labels
        self.latex_legend_label = self._create_legend_label()
        self.matrix_legend_label = self._create_legend_label()
        
        # Update colors and set initial text
        self.update_colors(self.text_color, self.border_color)

        # Add labels to layouts
        legends_layout.addWidget(self.latex_legend_label)
        legends_layout.addWidget(self.matrix_legend_label)
        layout.addLayout(legends_layout)
        self.setLayout(layout)

    def _create_legend_label(self):
        """Create a configured QLabel for legends."""
        label = QLabel()
        label.setWordWrap(True)
        label.setFont(QFont(self.FONT_FAMILY, self.FONT_SIZE))
        return label

    def update_theme(self, theme):
        """
        Update the window's theme.

        Args:
            theme (dict): New theme settings
        """
        self.text_color = theme["colors"]["text"]
        self.border_color = theme["colors"]["border"]
        self.background_color = theme["colors"]["background"]
        
        self.setStyleSheet(f"background-color: {self.background_color};")
        self.update_colors(self.text_color, self.border_color)

    def update_colors(self, text_color, border_color):
        """
        Update the colors of the legend text.

        Args:
            text_color (str): New text color
            border_color (str): New border color
        """
        self.text_color = text_color
        self.border_color = border_color
        
        table_style = (f"color: {text_color}; border-collapse: collapse; "
                      f"width: 100%; border: 2px solid {border_color};")
        
        latex_legend_text = f"<table style='{table_style}'>{SIMPLE_LATEX_LEGEND}</table>"
        matrix_legend_text = f"<table style='{table_style}'>{MATRIX_LEGEND}</table>"
        
        self.latex_legend_label.setText(latex_legend_text)
        self.matrix_legend_label.setText(matrix_legend_text)
