from PyQt5.QtWidgets import (
    QPushButton, QLabel, QComboBox, QHBoxLayout, QVBoxLayout, QMenu, QTextEdit,
    QSizePolicy, QInputDialog, QMessageBox, QScrollArea, QSpacerItem, QFrame
)
from PyQt5.QtGui import QFont, QTextOption, QIcon, QColor
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve
from themes.theme_manager import get_tokyo_night_theme, get_aura_theme, get_light_theme, get_anysphere_theme, ThemeManager
from ui.legend_window import LegendWindow
from ui.ui_config import UiConfig
from PyQt5.QtGui import QFont
from ui.visualization_window import VisualizationWindow
import json
import logging

logger = logging.getLogger(__name__)

class UIComponents:
    DEFAULT_FONT_FAMILY = "Menlo"
    DEFAULT_FONT_SIZE = 14
    
    def __init__(self, parent):
        """
        Initialize UIComponents with a reference to the parent CalculatorApp.
        
        Args:
            parent (CalculatorApp): The instance of the main calculator application.
        """
        self.parent = parent
        self.visualization_window = None
        self.theme_manager = ThemeManager()
        
        # Load theme settings
        self.current_theme = self.load_theme_settings()
        self.theme_colors = self.theme_manager.themes[self.current_theme]["colors"]
        
        # Load font settings
        font_settings = self.load_font_settings()
        font_family = font_settings.get('family', self.DEFAULT_FONT_FAMILY)
        font_size = font_settings.get('size', self.DEFAULT_FONT_SIZE)

        self.parent.PLACEHOLDER_TEXT = 'Enter Simplified LaTeX expression, e.g., 5C2 + sin(pi/2)\n' \
            'Or MATLAB expression, e.g., nchoosek(5,2) + sin(pi/2)'
        self.parent.FORMULA_FONT = QFont(font_family, font_size)

    def init_ui(self):
        """Initialize the calculator's user interface."""
        self._setup_window()
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)  # Increased spacing between components
        main_layout.setContentsMargins(20, 20, 20, 20)  # Add more padding around the edges
        
        layouts = {
            'top': self._create_top_buttons(),
            'mode': self._create_mode_selection(),
            'angle': self._create_angle_selection(),
            'matrix': self._create_matrix_components(),
            'formula': self._create_formula_components(),
            'result': self._create_result_components()
        }
        
        # Add a separator after the top buttons
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.HLine)
        separator1.setFrameShadow(QFrame.Sunken)
        separator1.setStyleSheet(f"background-color: rgba({self._get_rgba_from_hex(self.theme_colors['border'], 0.5)}); margin: 5px 0;")
        separator1.setMaximumHeight(1)
        separator1.setObjectName("separator1")
        self.parent.separator1 = separator1
        
        # Add a separator before the formula components
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setFrameShadow(QFrame.Sunken)
        separator2.setStyleSheet(f"background-color: rgba({self._get_rgba_from_hex(self.theme_colors['border'], 0.5)}); margin: 5px 0;")
        separator2.setMaximumHeight(1)
        separator2.setObjectName("separator2")
        self.parent.separator2 = separator2
        
        main_layout.addLayout(layouts['top'])
        main_layout.addWidget(separator1)
        
        # Create a container for mode and angle selection
        mode_angle_container = QHBoxLayout()
        mode_angle_container.addLayout(layouts['mode'])
        mode_angle_container.addSpacing(20)  # Add spacing between mode and angle
        mode_angle_container.addLayout(layouts['angle'])
        
        main_layout.addLayout(mode_angle_container)
        main_layout.addWidget(separator2)
        main_layout.addLayout(layouts['matrix'])
        main_layout.addLayout(layouts['formula'])
        main_layout.addLayout(layouts['result'])
        
        self.parent.setLayout(main_layout)

    def _setup_window(self):
        self.parent.setWindowTitle('VorTeX Calculator')
        self.parent.setGeometry(100, 100, 750, 550)  # Slightly larger window

    def _create_top_buttons(self):
        """Create and position buttons at the top right of the app."""
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 10)
        
        # Add title to the left
        title_label = QLabel('VorTeX Calculator')
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setStyleSheet(f"color: {self.theme_colors['button']};")
        title_label.setObjectName("title_label")
        self.parent.title_label = title_label
        
        layout.addWidget(title_label)
        layout.addStretch()
        
        # Create settings button with improved styling
        button_style = self._get_button_style()
        
        self.parent.settings_button = QPushButton('Settings')
        self.parent.settings_button.clicked.connect(self.parent.show_settings)
        self.parent.settings_button.setFixedSize(120, 40)
        self.parent.settings_button.setStyleSheet(button_style)
        self.parent.settings_button.setObjectName("settings_button")
        
        # Create legend button with improved styling
        self.parent.legend_button = QPushButton('Legend')
        self.parent.legend_button.clicked.connect(self.parent.show_legend)
        self.parent.legend_button.setFixedSize(120, 40)
        self.parent.legend_button.setStyleSheet(button_style)
        self.parent.legend_button.setObjectName("legend_button")
        
        # Add buttons to layout with spacing
        layout.addWidget(self.parent.settings_button)
        layout.addSpacing(10)
        layout.addWidget(self.parent.legend_button)
        
        return layout

    def _create_mode_selection(self):
        """Create input mode selection components."""
        mode_layout = QVBoxLayout()
        mode_layout.setSpacing(8)
        
        self.parent.label_mode = QLabel('Input Mode:')
        self.parent.label_mode.setFont(QFont("Arial", 13, QFont.Bold))
        self.parent.label_mode.setStyleSheet(f"color: {self.theme_colors['text']};")
        self.parent.label_mode.setObjectName("label_mode")
        
        # Load font settings
        font_settings = self.load_font_settings()
        font_family = font_settings.get('family', self.DEFAULT_FONT_FAMILY)
        font_size = font_settings.get('size', 12)  # Slightly smaller for UI elements
        
        self.parent.combo_mode = QComboBox()
        self.parent.combo_mode.addItems(['LaTeX', 'MATLAB', 'SymPy', 'Matrix'])
        self.parent.combo_mode.setFont(QFont(font_family, font_size))
        self.parent.combo_mode.setFixedWidth(180)
        self.parent.combo_mode.setFixedHeight(36)
        self.parent.combo_mode.setStyleSheet(self._get_combobox_style())
        self.parent.combo_mode.currentTextChanged.connect(self.parent.on_mode_changed)
        self.parent.combo_mode.setObjectName("combo_mode")
        
        mode_layout.addWidget(self.parent.label_mode)
        mode_layout.addWidget(self.parent.combo_mode)
        
        return mode_layout

    def _create_angle_selection(self):
        """Create angle mode selection components."""
        angle_layout = QVBoxLayout()
        angle_layout.setSpacing(8)
        
        self.parent.label_angle = QLabel('Angle Mode:')
        self.parent.label_angle.setFont(QFont("Arial", 13, QFont.Bold))
        self.parent.label_angle.setStyleSheet(f"color: {self.theme_colors['text']};")
        self.parent.label_angle.setObjectName("label_angle")
        
        # Load font settings
        font_settings = self.load_font_settings()
        font_family = font_settings.get('family', self.DEFAULT_FONT_FAMILY)
        font_size = font_settings.get('size', 12)  # Slightly smaller for UI elements
        
        self.parent.combo_angle = QComboBox()
        self.parent.combo_angle.addItems(['Degree', 'Radian'])
        self.parent.combo_angle.setFont(QFont(font_family, font_size))
        self.parent.combo_angle.setFixedWidth(180)
        self.parent.combo_angle.setFixedHeight(36)
        self.parent.combo_angle.setStyleSheet(self._get_combobox_style())
        self.parent.combo_angle.setObjectName("combo_angle")
        
        angle_layout.addWidget(self.parent.label_angle)
        angle_layout.addWidget(self.parent.combo_angle)
        
        return angle_layout
    

    def _create_matrix_components(self):
        matrix_layout = QVBoxLayout()
        matrix_layout.setSpacing(10)
        
        # Load font settings
        font_settings = self.load_font_settings()
        font_family = font_settings.get('family', self.DEFAULT_FONT_FAMILY)
        font_size = font_settings.get('size', self.DEFAULT_FONT_SIZE)
        
        self.parent.matrix_input = QTextEdit()
        self.parent.matrix_input.setPlaceholderText(
            "Enter matrix in MATLAB format, e.g., [1 2; 3 4]\n"
            "Or [1, 2; 3, 4] for comma-separated values"
        )
        self.parent.matrix_input.setFont(QFont(font_family, font_size))
        self.parent.matrix_input.setFixedHeight(120)  # Slightly taller
        self.parent.matrix_input.setStyleSheet(self._get_textedit_style())
        self.parent.matrix_input.hide()
        self.parent.matrix_input.setObjectName("matrix_input")
        
        operation_layout = QHBoxLayout()
        self.parent.label_matrix_op = QLabel('Matrix Operation:')
        self.parent.label_matrix_op.setFont(QFont("Arial", 13, QFont.Bold))
        self.parent.label_matrix_op.setStyleSheet(f"color: {self.theme_colors['text']};")
        self.parent.label_matrix_op.setFixedWidth(140)
        self.parent.label_matrix_op.setObjectName("label_matrix_op")
        
        self.parent.combo_matrix_op = QComboBox()
        self.parent.combo_matrix_op.addItems([
            'Determinant', 'Inverse', 'Eigenvalues', 'Rank',
            'Multiply', 'Add', 'Subtract', 'Divide', 'Differentiate'
        ])
        self.parent.combo_matrix_op.setFixedWidth(180)
        self.parent.combo_matrix_op.setFixedHeight(36)
        self.parent.combo_matrix_op.setStyleSheet(self._get_combobox_style())
        self.parent.combo_matrix_op.hide()
        self.parent.combo_matrix_op.setObjectName("combo_matrix_op")
        self.parent.label_matrix_op.hide()
        
        operation_layout.addWidget(self.parent.label_matrix_op)
        operation_layout.addWidget(self.parent.combo_matrix_op)
        operation_layout.addStretch()
        
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        button_style = self._get_button_style()
        
        self.parent.store_matrix_button = QPushButton('Store Matrix')
        self.parent.store_matrix_button.setFixedSize(130, 36)
        self.parent.store_matrix_button.clicked.connect(self.parent.store_matrix)
        self.parent.store_matrix_button.setStyleSheet(button_style)
        self.parent.store_matrix_button.hide()
        self.parent.store_matrix_button.setObjectName("store_matrix_button")
        
        self.parent.recall_matrix_button = QPushButton('Recall Matrix')
        self.parent.recall_matrix_button.setFixedSize(130, 36)
        self.parent.recall_matrix_button.clicked.connect(self.parent.recall_matrix)
        self.parent.recall_matrix_button.setStyleSheet(button_style)
        self.parent.recall_matrix_button.hide()
        self.parent.recall_matrix_button.setObjectName("recall_matrix_button")
        
        self.parent.calculate_matrix_button = QPushButton('Calculate Matrix')
        self.parent.calculate_matrix_button.setFixedSize(180, 36)
        self.parent.calculate_matrix_button.clicked.connect(self.parent.calculate_matrix)
        self.parent.calculate_matrix_button.setStyleSheet(button_style)
        self.parent.calculate_matrix_button.hide()
        self.parent.calculate_matrix_button.setObjectName("calculate_matrix_button")
        
        button_layout.addWidget(self.parent.store_matrix_button)
        button_layout.addWidget(self.parent.recall_matrix_button)
        button_layout.addWidget(self.parent.calculate_matrix_button)
        button_layout.addStretch()
        
        matrix_layout.addWidget(self.parent.matrix_input)
        matrix_layout.addLayout(operation_layout)
        matrix_layout.addLayout(button_layout)
        
        # Visibility managed by mode_config
        self.parent.store_matrix_button.hide()
        self.parent.recall_matrix_button.hide()
        self.parent.calculate_matrix_button.hide()
        return matrix_layout

    def _create_formula_components(self):
        """Create formula input components with vertical layout."""
        formula_layout = QVBoxLayout()
        formula_layout.setContentsMargins(0, 10, 0, 10)
        formula_layout.setSpacing(15)
        
        # Create header with label
        header_layout = QHBoxLayout()
        self.parent.label_formula = QLabel('Math Input:')
        self.parent.label_formula.setFont(QFont("Arial", 14, QFont.Bold))
        self.parent.label_formula.setStyleSheet(f"color: {self.theme_colors['text']};")
        self.parent.label_formula.setObjectName("label_formula")
        header_layout.addWidget(self.parent.label_formula)
        header_layout.addStretch()
        
        # Create input field with improved styling
        self.parent.entry_formula = QTextEdit()
        self.parent.entry_formula.setPlaceholderText(self.parent.PLACEHOLDER_TEXT)
        self.parent.entry_formula.setFont(self.parent.FORMULA_FONT)
        self.parent.entry_formula.setFixedHeight(120)  # Slightly taller
        self.parent.entry_formula.setStyleSheet(self._get_textedit_style())
        self.parent.entry_formula.setObjectName("entry_formula")
        
        # Create button layout
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)
        
        button_style = self._get_button_style()
        
        self.parent.calculate_button = QPushButton('Calculate')
        self.parent.calculate_button.clicked.connect(self.parent.calculate)
        self.parent.calculate_button.setFixedHeight(40)
        self.parent.calculate_button.setMinimumWidth(150)
        self.parent.calculate_button.setStyleSheet(button_style)
        self.parent.calculate_button.setObjectName("calculate_button")
        
        self.parent.visualize_button = QPushButton('Visualize')
        self.parent.visualize_button.clicked.connect(self.handle_visualization)
        self.parent.visualize_button.setFixedHeight(40)
        self.parent.visualize_button.setMinimumWidth(150)
        self.parent.visualize_button.setStyleSheet(button_style)
        self.parent.visualize_button.setObjectName("visualize_button")
        
        button_layout.addWidget(self.parent.calculate_button)
        button_layout.addWidget(self.parent.visualize_button)
        button_layout.addStretch()
        
        formula_layout.addLayout(header_layout)
        formula_layout.addWidget(self.parent.entry_formula)
        formula_layout.addLayout(button_layout)
        
        return formula_layout

    def _create_result_components(self):
        result_layout = QVBoxLayout()
        result_layout.setContentsMargins(0, 10, 0, 0)
        result_layout.setSpacing(10)
        
        # Create header with improved styling
        header_layout = QHBoxLayout()
        self.parent.result_label = QLabel('Result:')
        self.parent.result_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.parent.result_label.setStyleSheet(f"color: {self.theme_colors['text']};")
        self.parent.result_label.setObjectName("result_label")
        header_layout.addWidget(self.parent.result_label)
        header_layout.addStretch()
        
        # Use the same font as math input with fixed size of 14
        font_settings = self.load_font_settings()
        font_family = font_settings.get('family', self.DEFAULT_FONT_FAMILY)
        
        # Create result display with improved styling
        self.parent.result_display = QLabel()
        self.parent.result_display.setFont(QFont(font_family, 14))
        self.parent.result_display.setWordWrap(True)
        self.parent.result_display.setTextInteractionFlags(Qt.TextSelectableByMouse)  # Make text selectable
        self.parent.result_display.setMinimumHeight(100)
        self.parent.result_display.setStyleSheet(f"""
            QLabel {{
                padding: 15px;
                background-color: rgba({self._get_rgba_from_hex(self.theme_colors['background'], 0.7)});
                border: 1px solid {self.theme_colors['border']};
                border-radius: 5px;
            }}
        """)
        self.parent.result_display.setObjectName("result_display")
        
        result_layout.addLayout(header_layout)
        result_layout.addWidget(self.parent.result_display)
        
        return result_layout

    def handle_visualization(self):
        if self.visualization_window is None:
            self.visualization_window = VisualizationWindow(self.parent)
        
        self.visualization_window.show()
        self.parent.handle_visualization()

    def load_font_settings(self):
        """Load font settings from settings.json"""
        try:
            with open('settings.json', 'r') as f:
                settings = json.load(f)
                font_settings = settings.get('font_settings', {})
                
                # If font_settings exists but doesn't have family or size, add defaults
                if 'family' not in font_settings:
                    font_settings['family'] = self.DEFAULT_FONT_FAMILY
                if 'size' not in font_settings:
                    font_settings['size'] = self.DEFAULT_FONT_SIZE
                    
                return font_settings
        except FileNotFoundError:
            # Return default settings if file doesn't exist
            return {"family": self.DEFAULT_FONT_FAMILY, "size": self.DEFAULT_FONT_SIZE}
        except Exception as e:
            logger.error(f"Error loading font settings: {e}")
            return {"family": self.DEFAULT_FONT_FAMILY, "size": self.DEFAULT_FONT_SIZE}
    
    def load_theme_settings(self):
        """Load theme settings from settings.json"""
        try:
            with open('settings.json', 'r') as f:
                settings = json.load(f)
                theme = settings.get('theme', 'anysphere')
                return theme
        except FileNotFoundError:
            # Return default theme if file doesn't exist
            return "anysphere"
        except Exception as e:
            logger.error(f"Error loading theme settings: {e}")
            return "anysphere"
    
    def update_theme(self, theme_name):
        """Update the UI with the new theme"""
        self.current_theme = theme_name
        
        # Get theme colors and validate them
        try:
            self.theme_colors = self.theme_manager.themes[theme_name]["colors"]
            self._validate_theme_colors()
        except Exception as e:
            logger.error(f"Error updating theme: {e}")
            # Fall back to anysphere theme if there's an error
            self.current_theme = "anysphere"
            self.theme_colors = self.theme_manager.themes["anysphere"]["colors"]
            self._validate_theme_colors()
        
        # Update all UI components with the new theme
        self._update_ui_components()
    
    def _validate_theme_colors(self):
        """Ensure all theme colors are valid hex values"""
        required_colors = ['background', 'text', 'button', 'button_text', 'button_hover', 'border']
        default_colors = {
            'background': '#0e1116',
            'text': '#e6edf3',
            'button': '#E6C895',
            'button_text': '#ffffff',
            'button_hover': '#E6C895',
            'border': '#30363d',
            'input_bg': '#0e1116',
            'title': '#e6edf3'
        }
        
        # Validate each color and replace invalid ones with defaults
        for color_key in required_colors:
            if color_key not in self.theme_colors:
                logger.warning(f"Missing theme color: {color_key}, using default")
                self.theme_colors[color_key] = default_colors[color_key]
            else:
                color = self.theme_colors[color_key]
                if not isinstance(color, str) or not color.startswith('#') or len(color.lstrip('#')) != 6:
                    logger.warning(f"Invalid theme color for {color_key}: {color}, using default")
                    self.theme_colors[color_key] = default_colors[color_key]
        
        # Ensure input_bg exists
        if 'input_bg' not in self.theme_colors:
            self.theme_colors['input_bg'] = self.theme_colors['background']
    
    def _update_ui_components(self):
        """Update all UI components with the current theme colors"""
        # Update title
        if hasattr(self.parent, 'title_label'):
            self.parent.title_label.setStyleSheet(f"color: {self.theme_colors['button']};")
        
        # Update buttons
        button_style = self._get_button_style()
        for button_name in ['settings_button', 'legend_button', 'calculate_button', 'visualize_button', 
                           'store_matrix_button', 'recall_matrix_button', 'calculate_matrix_button']:
            if hasattr(self.parent, button_name):
                button = getattr(self.parent, button_name)
                button.setStyleSheet(button_style)
        
        # Update labels
        for label_name in ['label_mode', 'label_angle', 'label_formula', 'result_label', 'label_matrix_op']:
            if hasattr(self.parent, label_name):
                label = getattr(self.parent, label_name)
                label.setStyleSheet(f"color: {self.theme_colors['text']};")
        
        # Update comboboxes
        combobox_style = self._get_combobox_style()
        for combo_name in ['combo_mode', 'combo_angle', 'combo_matrix_op']:
            if hasattr(self.parent, combo_name):
                combo = getattr(self.parent, combo_name)
                combo.setStyleSheet(combobox_style)
        
        # Update text edits
        textedit_style = self._get_textedit_style()
        for edit_name in ['entry_formula', 'matrix_input']:
            if hasattr(self.parent, edit_name):
                edit = getattr(self.parent, edit_name)
                edit.setStyleSheet(textedit_style)
        
        # Update result display
        if hasattr(self.parent, 'result_display'):
            self.parent.result_display.setStyleSheet(f"""
                QLabel {{
                    padding: 15px;
                    background-color: rgba({self._get_rgba_from_hex(self.theme_colors['background'], 0.7)});
                    border: 1px solid {self.theme_colors['border']};
                    border-radius: 5px;
                }}
            """)
        
        # Update separators
        for sep_name in ['separator1', 'separator2']:
            if hasattr(self.parent, sep_name):
                sep = getattr(self.parent, sep_name)
                sep.setStyleSheet(f"background-color: rgba({self._get_rgba_from_hex(self.theme_colors['border'], 0.5)}); margin: 5px 0;")
    
    def _get_button_style(self):
        """Generate button style based on current theme"""
        button_color = self.theme_colors['button']
        button_text_color = self.theme_colors['button_text']
        
        # For light theme, use a darker text color for better contrast
        if self.current_theme == 'light':
            button_text_color = "#333333"
            
        button_hover = self.theme_colors['button_hover']
        
        return f"""
            QPushButton {{
                background-color: rgba({self._get_rgba_from_hex(button_color, 0.1)});
                border: 1px solid {button_color};
                border-radius: 5px;
                color: {button_text_color};
                font-weight: bold;
                padding: 5px 15px;
            }}
            QPushButton:hover {{
                background-color: rgba({self._get_rgba_from_hex(button_hover, 0.2)});
                border: 1px solid {button_hover};
            }}
            QPushButton:pressed {{
                background-color: rgba({self._get_rgba_from_hex(button_hover, 0.3)});
            }}
        """
    
    def _get_combobox_style(self):
        """Generate combobox style based on current theme"""
        border_color = self.theme_colors['border']
        button_color = self.theme_colors['button']
        background_color = self.theme_colors['input_bg'] if 'input_bg' in self.theme_colors else self.theme_colors['background']
        
        return f"""
            QComboBox {{
                border: 1px solid {border_color};
                border-radius: 5px;
                padding: 5px 10px;
                background-color: rgba({self._get_rgba_from_hex(background_color, 0.7)});
                color: {self.theme_colors['text']};
            }}
            QComboBox:hover {{
                border: 1px solid {button_color};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
        """
    
    def _get_textedit_style(self):
        """Generate textedit style based on current theme"""
        border_color = self.theme_colors['border']
        button_color = self.theme_colors['button']
        background_color = self.theme_colors['input_bg'] if 'input_bg' in self.theme_colors else self.theme_colors['background']
        
        return f"""
            QTextEdit {{
                border: 1px solid {border_color};
                border-radius: 5px;
                padding: 10px;
                background-color: rgba({self._get_rgba_from_hex(background_color, 0.7)});
                color: {self.theme_colors['text']};
            }}
            QTextEdit:focus {{
                border: 1px solid {button_color};
            }}
        """
    
    def _get_rgba_from_hex(self, hex_color, alpha=1.0):
        """Convert hex color to rgba format for QSS"""
        try:
            # Check if the color is a valid hex color
            if not hex_color or not isinstance(hex_color, str) or not hex_color.startswith('#'):
                # Default to a dark gray if invalid
                hex_color = "#333333"
            
            hex_color = hex_color.lstrip('#')
            
            # Ensure we have a valid hex color (6 characters)
            if len(hex_color) != 6:
                hex_color = "333333"  # Default to dark gray
                
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return f"{r}, {g}, {b}, {alpha}"
        except ValueError:
            # If there's any error, return a default color
            return f"51, 51, 51, {alpha}"  # Dark gray with specified alpha
