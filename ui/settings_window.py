from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QComboBox, QPushButton, QGroupBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from themes.theme_manager import ThemeManager, get_aura_theme, get_tokyo_night_theme, get_light_theme
import json
import logging

logger = logging.getLogger(__name__)

class SettingsWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.theme_manager = ThemeManager()
        
        # Load saved theme
        saved_theme = self.load_settings()
        if hasattr(self.parent, 'current_theme'):
            self.parent.current_theme = saved_theme
        
        self.init_ui()
        self.apply_theme()

    def init_ui(self):
        self.setWindowTitle('Settings')
        self.setFixedSize(450, 300)
        
        # Main layout with padding
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Create appearance group
        appearance_group = QGroupBox("Appearance")
        font = QFont("Arial", 13)
        font.setBold(True)
        appearance_group.setFont(font)
        appearance_group.setStyleSheet("""
            QGroupBox {
                margin-top: 1.5em;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        appearance_layout = QVBoxLayout()
        appearance_layout.setSpacing(10)
        appearance_layout.setContentsMargins(15, 15, 15, 15)
        
        # Theme selection
        theme_layout = QHBoxLayout()
        theme_label = QLabel("Theme:")
        theme_label.setFont(QFont("Arial", 13))
        
        self.theme_combo = QComboBox()
        self.theme_combo.setFont(QFont("Monaspace Neon", 12))
        self.theme_combo.addItems(["Tokyo night", "Aura", "Light"])
        self.theme_combo.setFixedHeight(30)
        
        # Set current theme
        current_theme = self.parent.current_theme if hasattr(self.parent, 'current_theme') else "aura"
        index = self.theme_combo.findText(current_theme)
        if index >= 0:
            self.theme_combo.setCurrentIndex(index)
            
        self.theme_combo.currentTextChanged.connect(self.on_theme_changed)
        
        theme_layout.addWidget(theme_label)
        theme_layout.addWidget(self.theme_combo)
        theme_layout.addStretch()
        
        appearance_layout.addLayout(theme_layout)
        appearance_group.setLayout(appearance_layout)
        
        # Add groups to main layout
        layout.addWidget(appearance_group)
        
        # Add stretch to push everything to the top
        layout.addStretch()
        
        # Create button row
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        self.save_button = QPushButton("Save")
        self.save_button.setFont(QFont("Arial", 13))
        self.save_button.setFixedSize(100, 35)
        self.save_button.clicked.connect(self.save_settings)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setFont(QFont("Arial", 13))
        self.cancel_button.setFixedSize(100, 35)
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)

    def apply_theme(self):
        """Apply theme to the settings window"""
        current_theme = self.parent.current_theme
        
        if current_theme == "tokyo_night":
            theme_data = get_tokyo_night_theme()
        elif current_theme == "aura":
            theme_data = get_aura_theme()
        elif current_theme == "light":
            theme_data = get_light_theme()
        else:
            theme_data = get_aura_theme()
        
        # Apply theme to the settings window
        self.setStyleSheet(theme_data["main_widget"])
        
        # Apply theme to specific components
        for widget in self.findChildren(QPushButton):
            widget.setStyleSheet(theme_data["theme_button"])
            
        for widget in self.findChildren(QComboBox):
            widget.setStyleSheet(theme_data["combo_box"])
            
        for widget in self.findChildren(QLabel):
            widget.setStyleSheet(theme_data["label"])
            
        # Maintain consistent sizing
        self.setFixedSize(self.sizeHint())

    def on_theme_changed(self, theme_name):
        """Handle theme change event"""
        # Convert theme names to the format expected by the theme manager
        theme_map = {
            "Tokyo night": "tokyo_night",
            "Aura": "aura",
            "Light": "light"
        }
        
        theme_key = theme_map.get(theme_name, "aura")
        
        if self.parent:
            # Update the parent's current theme
            self.parent.current_theme = theme_key
            # Apply the theme to the parent window
            self.parent.set_theme(theme_key)
            # Apply theme to settings window
            self.apply_theme()
            # Save the theme preference (optional)
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
