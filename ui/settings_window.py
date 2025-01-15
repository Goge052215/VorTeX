from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QComboBox, QPushButton, QGroupBox, QWidget
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from themes.theme_manager import ThemeManager, get_aura_theme, get_tokyo_night_theme, get_light_theme
import json
import logging

logger = logging.getLogger(__name__)

class SettingsWindow(QDialog):
    WINDOW_WIDTH = 600
    WINDOW_HEIGHT = 400
    FONT_FAMILY = "Arial"
    FONT_SIZE = 13
    GROUP_BOX_STYLE = """
        QGroupBox {
            margin-top: 1.5em;
            padding: 15px;
            background-color: transparent;
            border-radius: 8px;
            border: 1px solid #6d6d6d;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
            color: #edecee;
        }
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.theme_manager = ThemeManager()
        self.init_settings()
        self.init_ui()
        self.apply_theme()

    def init_settings(self):
        """Initialize settings from saved configuration"""
        try:
            with open('settings.json', 'r') as f:
                settings = json.load(f)
                saved_theme = settings.get('theme', 'aura')
                if hasattr(self.parent, 'current_theme'):
                    self.parent.current_theme = saved_theme
        except (FileNotFoundError, json.JSONDecodeError):
            if hasattr(self.parent, 'current_theme'):
                self.parent.current_theme = 'aura'

    def init_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout()
        
        self._create_appearance_group(layout)
        self._create_about_group(layout)  # Add the about group
        
        self.setLayout(layout)
        self.setWindowTitle("Settings")
        self.setFixedSize(self.WINDOW_WIDTH, self.WINDOW_HEIGHT)

    def _setup_window(self):
        """Configure basic window properties"""
        self.setWindowTitle('Settings')
        self.setFixedSize(self.WINDOW_WIDTH, self.WINDOW_HEIGHT)

    def _create_main_layout(self):
        """Create and configure the main layout"""
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        return layout

    def _get_theme_specific_styles(self, theme_name):
        """Get theme-specific styles based on the current theme"""
        if theme_name in ["aura", "tokyo_night"]:
            return {
                "group_box": """
                    QGroupBox {
                        margin-top: 1.5em;
                        padding: 15px;
                        background-color: #1c1b22;
                        border-radius: 8px;
                    }
                    QGroupBox::title {
                        subcontrol-origin: margin;
                        left: 10px;
                        padding: 0 5px;
                        color: #edecee;
                    }
                """,
                "label_color": "color: #edecee;",
                "button_style": """
                    QPushButton {
                        background-color: #4a90e2;
                        color: white;
                        border: none;
                        border-radius: 8px;
                        padding: 8px;
                    }
                    QPushButton:hover {
                        background-color: #357abd;
                    }
                """
            }
        else:  # Light theme
            return {
                "group_box": """
                    QGroupBox {
                        margin-top: 1.5em;
                        padding: 15px;
                        background-color: #e6e6e6;
                        border-radius: 8px;
                    }
                    QGroupBox::title {
                        subcontrol-origin: margin;
                        left: 10px;
                        padding: 0 5px;
                        color: #333333;
                    }
                """,
                "label_color": "color: #333333;",
                "button_style": """
                    QPushButton {
                        background-color: #4a90e2;
                        color: white;
                        border: none;
                        border-radius: 8px;
                        padding: 8px;
                    }
                    QPushButton:hover {
                        background-color: #357abd;
                    }
                """
            }

    def _create_appearance_group(self, layout):
        """Create and configure the appearance settings group"""
        appearance_group = QGroupBox("Appearance")
        appearance_group.setFont(QFont(self.FONT_FAMILY, self.FONT_SIZE, QFont.Bold))
        
        # Get current theme
        current_theme = getattr(self.parent, 'current_theme', 'aura')
        
        # Add theme-specific background color and improved shape
        group_style = """
            QGroupBox {
                margin-top: 1.5em;
                padding: 20px;
                background-color: #1c1b22;
                border-radius: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #edecee;
            }
        """
        if current_theme == "light":
            group_style = """
                QGroupBox {
                    margin-top: 1.5em;
                    padding: 20px;
                    background-color: #e6e6e6;
                    border-radius: 12px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px;
                    color: #333333;
                }
            """
        
        appearance_group.setStyleSheet(group_style)
        appearance_layout = QVBoxLayout()
        appearance_layout.setSpacing(10)
        appearance_layout.setContentsMargins(15, 15, 15, 15)

        theme_layout = self._create_theme_selection()
        appearance_layout.addLayout(theme_layout)
        appearance_group.setLayout(appearance_layout)
        layout.addWidget(appearance_group)
        layout.addStretch()

    def _create_theme_selection(self, label_style=None):
        """Create the theme selection combo box and label"""
        theme_layout = QHBoxLayout()
        
        theme_label = QLabel("Theme:")
        theme_label.setFont(QFont("Monaspace Neon", 12))
        
        self.theme_combo = QComboBox()
        self.theme_combo.setFont(QFont("Monaspace Neon", 12))
        theme_display_names = {
            "tokyo_night": "Tokyo night",
            "aura": "Aura",
            "light": "Light"
        }
        self.theme_combo.addItems(list(theme_display_names.values()))
        self.theme_combo.setFixedHeight(35)
        self.theme_combo.setFixedWidth(200)
        
        display_name = theme_display_names.get('aura', "Aura")
        index = self.theme_combo.findText(display_name, Qt.MatchExactly)
        if index >= 0:
            self.theme_combo.setCurrentIndex(index)
        
        self.theme_combo.currentTextChanged.connect(self.on_theme_changed)
        
        theme_layout.addWidget(theme_label)
        theme_layout.addWidget(self.theme_combo)
        theme_layout.addStretch()
        
        return theme_layout

    def _create_button_row(self, layout):
        """Create the save and cancel buttons"""
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)

        current_theme = getattr(self.parent, 'current_theme', 'aura')
        styles = self._get_theme_specific_styles(current_theme)

        for button_text, callback in [("Save", self.save_settings), ("Cancel", self.reject)]:
            button = QPushButton(button_text)
            button.setFont(QFont(self.FONT_FAMILY, self.FONT_SIZE))
            button.setFixedSize(150, 40)
            button.setStyleSheet(styles["button_style"])
            button.clicked.connect(callback)
            button_layout.addWidget(button)

        button_layout.insertStretch(0, 1)
        layout.addLayout(button_layout)

    def apply_theme(self):
        """Apply theme to the settings window"""
        current_theme = getattr(self.parent, 'current_theme', 'aura')
        theme_data = get_aura_theme()
        
        if current_theme == "tokyo_night":
            theme_data = get_tokyo_night_theme()
        elif current_theme == "light":
            theme_data = get_light_theme()
        else:
            theme_data = get_aura_theme()
        
        # Apply theme to the settings window
        self.setStyleSheet(theme_data["main_widget"])
        
        # Apply theme to specific components
        for widget in self.findChildren(QPushButton):
            widget.setStyleSheet(theme_data["theme_button"])
        
        # Use main_widget style for ComboBox and Labels
        for widget in self.findChildren((QComboBox, QLabel)):
            widget.setStyleSheet(theme_data["main_widget"])
        
        # Update appearance group background based on theme
        for group_box in self.findChildren(QGroupBox):
            if current_theme == "light":
                group_box.setStyleSheet("""
                    QGroupBox {
                        margin-top: 1.5em;
                        padding: 15px;
                        background-color: #e6e6e6;
                        border-radius: 8px;
                    }
                    QGroupBox::title {
                        subcontrol-origin: margin;
                        left: 10px;
                        padding: 0 5px;
                        color: #333333;
                    }
                """)
            else:
                group_box.setStyleSheet(self.GROUP_BOX_STYLE)
        
        # Maintain consistent sizing
        self.setFixedSize(self.sizeHint())

    def on_theme_changed(self, theme_name):
        """Handle theme change event"""
        # Convert display names to internal theme names
        theme_map = {
            "Tokyo night": "tokyo_night",
            "Aura": "aura",
            "Light": "light"
        }
        
        theme_key = theme_map.get(theme_name, "aura")
        
        if self.parent:
            self.parent.current_theme = theme_key
            self.parent.set_theme(theme_key)
            self.apply_theme()
            self.save_theme_preference(theme_key)

    def save_settings(self):
        if self.parent:
            self.parent.current_theme = self.theme_combo.currentText()
        self.accept()

    def save_theme_preference(self, theme):
        """Save the theme preference to settings.json"""
        try:
            # Default settings if file doesn't exist
            settings = {
                "theme": theme,
                "input_mode": "LaTeX",
                "angle_mode": "Degree"
            }
            
            try:
                # Try to read existing settings
                with open('settings.json', 'r') as f:
                    existing_settings = json.load(f)
                    settings.update(existing_settings)
            except FileNotFoundError:
                # If file doesn't exist, we'll create it with default settings
                pass
            
            # Update theme and write all settings
            settings['theme'] = theme
            with open('settings.json', 'w') as f:
                json.dump(settings, f, indent=4)
                
        except Exception as e:
            logger.error(f"Error saving theme preference: {e}")

    def load_settings(self):
        """Load settings from settings.json"""
        try:
            with open('settings.json', 'r') as f:
                settings = json.load(f)
                return settings.get('theme', 'aura')  # Return 'aura' if theme not found
        except FileNotFoundError:
            # Create default settings file if it doesn't exist
            default_settings = {
                "theme": "aura",
                "input_mode": "LaTeX",
                "angle_mode": "Degree"
            }
            with open('settings.json', 'w') as f:
                json.dump(default_settings, f, indent=4)
            return "aura"
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            return "aura"

    def _create_about_group(self, layout):
        """Create and configure the about section"""
        about_group = QGroupBox("About VorTeX")
        about_group.setFont(QFont(self.FONT_FAMILY, self.FONT_SIZE, QFont.Bold))
        
        # Style for transparent background with border
        about_style = """
            QGroupBox {
                margin-top: 1.5em;
                padding: 20px;
                background-color: transparent;
                border: 1px solid #edecee;
                border-radius: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #edecee;
            }
        """
        about_group.setStyleSheet(about_style)
        
        about_layout = QVBoxLayout()
        about_layout.setSpacing(10)
        about_layout.setContentsMargins(15, 15, 15, 15)
        
        about_label = QLabel("VorTeX Calculator\nVersion 1.0.1\nCopyright © 2025, George Huang, All rights reserved.\nGitHub: https://github.com/Goge052215/VorTeX")
        about_label.setFont(QFont("Monaspace Neon", 12))
        about_layout.addWidget(about_label)
        
        about_group.setLayout(about_layout)
        layout.addWidget(about_group)
        layout.addStretch()
