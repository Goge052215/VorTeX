from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QComboBox, QPushButton, QGroupBox, QWidget,
    QMessageBox, QTextEdit, QListWidget, QStackedWidget,
    QFrame, QSplitter, QListWidgetItem, QFileDialog
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QFontDatabase, QIcon
from themes.theme_manager import ThemeManager, get_aura_theme, get_tokyo_night_theme, get_light_theme, get_anysphere_theme
import json
import logging
import matlab.engine
import os
import sys

logger = logging.getLogger(__name__)

class SettingsWindow(QDialog):
    WINDOW_WIDTH = 750
    WINDOW_HEIGHT = 550
    FONT_FAMILY = "Menlo"
    DISPLAY_FONT = ""
    FONT_SIZE = 14
    GROUP_BOX_STYLE = """
        QGroupBox {
            margin-top: 1.5em;
            padding: 15px;
            background-color: transparent;
            border-radius: 8px;
            border: none;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
            color: #e6edf3;
        }
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        print("SettingsWindow initialization started")
        self.parent = parent
        self.theme_manager = ThemeManager()
        print("Theme manager initialized")
        self.check_font_availability()
        print("Font availability checked")
        self.init_settings()
        print("Settings initialized")
        self.init_ui()
        print("UI initialized")
        self.check_license_validity()
        print("License validity checked")
        self.apply_theme()
        print("Theme applied")

    def check_font_availability(self):
        """Check if Monaspace Neon is installed and set default font accordingly"""
        print("Checking font availability...")
        font_db = QFontDatabase()
        available_fonts = [font.lower() for font in font_db.families()]
        
        if "monaspace neon" in available_fonts:
            print("Monaspace Neon is installed, setting as default")
            self.FONT_FAMILY = "Monaspace Neon"
        else:
            print(f"Monaspace Neon not found, using Menlo as fallback. Available fonts: {len(available_fonts)}")
            self.FONT_FAMILY = "Menlo"
        
        current_settings = self.load_font_settings()
        if 'family' not in current_settings:
            self.save_font_settings({'family': self.FONT_FAMILY})
            print(f"Default font set to {self.FONT_FAMILY}")

    def init_settings(self):
        """Initialize settings from saved configuration"""
        default_theme = 'anysphere'
        
        if hasattr(self.parent, 'current_theme') and self.parent.current_theme:
            print(f"Using parent's current theme: {self.parent.current_theme}")
            return
            
        try:
            with open('settings.json', 'r') as f:
                settings = json.load(f)
                saved_theme = settings.get('theme', default_theme)
                if hasattr(self.parent, 'current_theme'):
                    self.parent.current_theme = saved_theme
            print(f"Loaded settings with theme: {saved_theme}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Settings file not found or invalid: {e}")
            if hasattr(self.parent, 'current_theme'):
                self.parent.current_theme = default_theme
            self.save_theme_preference(default_theme)

    def init_ui(self):
        """Initialize the UI components with Obsidian-like layout"""
        print("Starting UI initialization")
        
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet("QSplitter { border: none; } QSplitter::handle { width: 1px; background-color: #4a4f57; }")
        
        # Left sidebar
        sidebar_widget = QWidget()
        sidebar_widget.setObjectName("sidebarWidget")
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(10, 15, 10, 0)
        sidebar_layout.setSpacing(5)
        
        sidebar_layout.addSpacing(10)
        
        # Navigation list with icons
        self.nav_list = QListWidget()
        self.nav_list.setObjectName("navList")
        
        # Create items without icons
        appearance_item = QListWidgetItem("Appearance")
        appearance_item.setFont(QFont(self.DISPLAY_FONT, 13))
        
        font_item = QListWidgetItem("Font Settings")
        font_item.setFont(QFont(self.DISPLAY_FONT, 13))
        
        matlab_item = QListWidgetItem("MATLAB License")
        matlab_item.setFont(QFont(self.DISPLAY_FONT, 13))
        
        logs_item = QListWidgetItem("Debug Logs")
        logs_item.setFont(QFont(self.DISPLAY_FONT, 13))
        
        about_item = QListWidgetItem("About")
        about_item.setFont(QFont(self.DISPLAY_FONT, 13))
        
        # Add items to list
        self.nav_list.addItem(appearance_item)
        self.nav_list.addItem(font_item)
        self.nav_list.addItem(matlab_item)
        self.nav_list.addItem(logs_item)
        self.nav_list.addItem(about_item)
        
        # Configure list item height (decreased)
        self.nav_list.setStyleSheet("""
            QListWidget::item {
                height: 28px;
                padding-left: 10px;
            }
        """)
        
        # Select the first item by default
        self.nav_list.setCurrentRow(0)
        
        self.nav_list.setFixedWidth(180)
        self.nav_list.currentRowChanged.connect(self.change_page)
        sidebar_layout.addWidget(self.nav_list)
        
        # Right content area
        content_widget = QWidget()
        content_widget.setObjectName("contentWidget")
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(30, 10, 30, 30)
        
        # Stacked widget to hold different pages
        self.pages = QStackedWidget()
        
        # Create pages
        appearance_page = QWidget()
        appearance_page.setObjectName("appearancePage")
        appearance_layout = QVBoxLayout(appearance_page)
        appearance_layout.setContentsMargins(0, 0, 0, 0)
        appearance_group = self._create_appearance_group()
        appearance_layout.addWidget(appearance_group)
        
        appearance_layout.addStretch()
        
        font_page = QWidget()
        font_page.setObjectName("fontPage")
        font_layout = QVBoxLayout(font_page)
        font_layout.setContentsMargins(0, 0, 0, 0)
        font_group = self._create_font_settings_group()
        font_layout.addWidget(font_group)
        font_layout.addStretch()
        
        matlab_page = QWidget()
        matlab_page.setObjectName("matlabPage")
        matlab_layout = QVBoxLayout(matlab_page)
        matlab_layout.setContentsMargins(0, 0, 0, 0)
        credentials_group = self._create_matlab_credentials()
        matlab_layout.addWidget(credentials_group)
        matlab_layout.addStretch()
        
        logs_page = QWidget()
        logs_page.setObjectName("logsPage")
        logs_layout = QVBoxLayout(logs_page)
        logs_layout.setContentsMargins(0, 0, 0, 0)
        logs_group = self._create_logs_group()
        logs_layout.addWidget(logs_group)
        logs_layout.addStretch()
        
        about_page = QWidget()
        about_page.setObjectName("aboutPage")
        about_layout = QVBoxLayout(about_page)
        about_layout.setContentsMargins(0, 0, 0, 0)
        about_group = self._create_about_group()
        about_layout.addWidget(about_group)
        about_layout.addStretch()
        
        # Add pages to stacked widget
        self.pages.addWidget(appearance_page)
        self.pages.addWidget(font_page)
        self.pages.addWidget(matlab_page)
        self.pages.addWidget(logs_page)
        self.pages.addWidget(about_page)
        
        content_layout.addWidget(self.pages)
        
        # Add widgets to splitter
        splitter.addWidget(sidebar_widget)
        splitter.addWidget(content_widget)
        
        # Set initial splitter sizes
        splitter.setSizes([200, 600])
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
        
        # Apply styles directly to make sure they take effect
        current_theme = getattr(self.parent, 'current_theme', 'anysphere')
        if current_theme == 'anysphere':
            # Dark theme colors
            sidebar_color = "#121417"  # Very dark grey/almost black for sidebar  
            content_color = "#1A1D21"  # Dark grey for content area
            
            # Apply directly to widgets
            self.setStyleSheet(f"background-color: {content_color};")
            sidebar_widget.setStyleSheet(f"background-color: {sidebar_color};")
            content_widget.setStyleSheet(f"background-color: {content_color};")
            self.nav_list.setStyleSheet(f"background-color: {sidebar_color}; border: none;")
            
        self.setLayout(main_layout)
        self.setWindowTitle("Settings")
        self.setFixedSize(self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
        print(f"Window size set: {self.width()}x{self.height()}")
        print("UI initialization completed")

    def change_page(self, index):
        """Change the current page in the stacked widget"""
        self.pages.setCurrentIndex(index)

    def _create_appearance_group(self):
        """Create the appearance settings group"""
        group_box = QGroupBox()
        group_box.setObjectName("appearanceGroup")
        
        # Set style based on current theme
        current_theme = getattr(self.parent, 'current_theme', 'anysphere')
        if current_theme == "light":
            appearance_style = """
                    QGroupBox {
                        padding: 15px 20px 20px 20px;
                        background-color: #f0f0f0;
                        border: none;
                        border-radius: 8px;
                    }
                    QLabel {
                        color: #333333;
                    }
                    QComboBox {
                        background-color: #ffffff;
                        color: #333333;
                        border: 1px solid #cccccc;
                    }
            """
        else:  # anysphere/dark theme
            appearance_style = """
                QGroupBox {
                    padding: 15px 20px 20px 20px;
                    background-color: #1A1D21;
                    border: none;
                    border-radius: 8px;
                }
                QLabel {
                    color: #e6edf3;
                }
                QComboBox {
                    background-color: #121417;
                    color: #e6edf3;
                    border: 1px solid #30363d;
                }
            """
        
        group_box.setStyleSheet(appearance_style)
        
        main_layout = QVBoxLayout()
        
        # Title directly in main layout
        title_label = QLabel("Appearance")
        title_label.setFont(QFont(self.DISPLAY_FONT, 14, QFont.Bold))
        main_layout.addWidget(title_label)
        
        # Add spacing after title
        main_layout.addSpacing(10)
        
        appearance_layout = QVBoxLayout()
        appearance_layout.setSpacing(10)

        # Theme selection
        theme_layout = self._create_theme_selection()
        appearance_layout.addLayout(theme_layout)
        
        main_layout.addLayout(appearance_layout)
        group_box.setLayout(main_layout)
        
        return group_box

    def _create_theme_selection(self, label_style=None):
        """Create the theme selection combo box and label"""
        theme_layout = QVBoxLayout()
        
        theme_description = QLabel(
            "Choose your preferred visual theme for the calculator interface."
        )
        theme_description.setFont(QFont(self.DISPLAY_FONT, 12.5))
        theme_description.setWordWrap(True)
        theme_description.setStyleSheet("margin-bottom: 20px;")
        theme_layout.addWidget(theme_description)
        
        theme_layout.addSpacing(10)
        
        theme_selector = QHBoxLayout()
        theme_label = QLabel("Theme:")
        theme_label.setFont(QFont(self.DISPLAY_FONT, 13))
        
        self.theme_combo = QComboBox()
        self.theme_combo.setFont(QFont(self.FONT_FAMILY, 12))
        theme_display_names = {
            "tokyo_night": "Tokyo night",
            "aura": "Aura",
            "light": "Light",
            "anysphere": "Anysphere"
        }
        self.theme_combo.addItems(list(theme_display_names.values()))
        self.theme_combo.setFixedHeight(28)
        self.theme_combo.setMinimumWidth(200)
        
        current_theme = getattr(self.parent, 'current_theme', 'anysphere')
        display_name = theme_display_names.get(current_theme, "Anysphere")
        index = self.theme_combo.findText(display_name, Qt.MatchExactly)
        if index >= 0:
            self.theme_combo.setCurrentIndex(index)
        
        self.theme_combo.currentTextChanged.connect(self.on_theme_changed)
        
        theme_selector.addWidget(theme_label)
        theme_selector.addWidget(self.theme_combo)
        theme_selector.addStretch()
        
        theme_layout.addLayout(theme_selector)
        
        return theme_layout

    def _create_button_row(self, layout):
        """Create the save and cancel buttons"""
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)

        current_theme = getattr(self.parent, 'current_theme', 'anysphere')
        styles = self._get_theme_specific_styles(current_theme)

        for button_text, callback in [("Save", self.save_settings), ("Cancel", self.reject)]:
            button = QPushButton(button_text)
            button.setFont(QFont(self.FONT_FAMILY, self.FONT_SIZE))
            button.setFixedSize(150, 32)  # Reduced height from 40 to 32
            button.setStyleSheet(styles["button_style"])
            button.clicked.connect(callback)
            button_layout.addWidget(button)

        button_layout.insertStretch(0, 1)
        layout.addLayout(button_layout)

    def apply_theme(self):
        """Apply the current theme to all UI components"""
        # Get current theme
        current_theme = getattr(self.parent, 'current_theme', 'anysphere')
        
        # Validate theme - ensure it's one of the available themes
        valid_themes = ["light", "tokyo_night", "aura", "anysphere"]
        if current_theme not in valid_themes:
            current_theme = 'anysphere'
            if hasattr(self.parent, 'current_theme'):
                self.parent.current_theme = current_theme
        
        # Get theme-specific accent colors
        accent_colors = {
            "light": "#4a90e2",     # Blue accent for light theme
            "tokyo_night": "#2ac3de", # Cyan accent for Tokyo Night
            "aura": "#a277ff",      # Purple accent for Aura
            "anysphere": "#E6C895"  # Gold accent for Anysphere (default)
        }
        
        # Get the accent color for the current theme
        accent_color = accent_colors.get(current_theme, "#E6C895")
        
        # Define sidebar and content colors based on theme
        sidebar_color = "#121417"  # Default dark sidebar
        content_color = "#1A1D21"  # Default dark content
        
        # Theme-specific colors
        if current_theme == "light":
            sidebar_color = "#e0e0e0" 
            content_color = "#f5f5f5"
        elif current_theme == "tokyo_night":
            sidebar_color = "#16161e"  # Darker sidebar for Tokyo Night
            content_color = "#1a1b26"  # Content area color
        elif current_theme == "aura":
            sidebar_color = "#12111a"  # Darker sidebar for Aura
            content_color = "#15141b"  # Content area color
        
        # Define theme-specific styles
        if current_theme == "light":
            # Light theme styles
            button_style = f"""
                QPushButton {{
                    background-color: #F7D08A;
                    border: none;
                    border-radius: 5px;
                    padding: 4px 10px;
                    color: #333333;
                    min-height: 26px;
                }}
                QPushButton:hover {{
                    background-color: #E8C27D;
                }}
                QPushButton:pressed {{
                    background-color: #D9B36C;
                }}
            """
            
            combo_style = f"""
                QComboBox {{
                    background-color: #ffffff;
                    border: 1px solid #c0c0c0;
                    border-radius: 5px;
                    padding: 4px 4px 4px 15px;
                    color: #333333;
                    min-height: 26px;
                }}
                QComboBox:hover {{
                    border: 1px solid {accent_color};
                }}
                QComboBox QAbstractItemView {{
                    background-color: #ffffff;
                    color: #333333;
                    selection-background-color: {accent_color};
                }}
            """
            
            label_style = """
                QLabel {
                    color: #333333;
                    font-size: 13.5px;
                    background-color: transparent;
                }
            """
            
            group_style = f"""
                QGroupBox {{
                    margin-top: 1.5em;
                    padding: 20px;
                    background-color: {content_color};
                    border: none;
                    border-radius: 8px;
                }}
                QGroupBox::title {{
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                    color: #333333;
                }}
            """
            
            text_edit_style = f"""
                QTextEdit {{
                    background-color: #ffffff;
                    border: 1px solid #c0c0c0;
                    border-radius: 5px;
                    color: #333333;
                }}
            """
            
            list_widget_style = f"""
                QListWidget {{
                    background-color: {sidebar_color};
                    border-right: none;
                    border-left: none;
                    border-top: none;
                    color: #333333;
                    padding: 10px 0px 0px 0px;
                }}
                QListWidget::item {{
                    padding: 6px 8px 6px 30px;
                    border-radius: 4px;
                    margin: 2px 8px 2px 8px;
                }}
                QListWidget::item:selected {{
                    background-color: {accent_color};
                    color: #ffffff;
                    font-weight: bold;
                }}
                QListWidget::item:hover {{
                    background-color: rgba(74, 144, 226, 0.3);
                    border: 1px solid rgba(74, 144, 226, 0.5);
                }}
            """
            
            splitter_style = """
                QSplitter::handle {
                    background-color: #dddddd;
                    width: 1px;
                }
            """
            
            main_style = f"""
                QDialog {{
                    background-color: {content_color};
                }}
                QStackedWidget {{
                    background-color: {content_color};
                }}
                #contentWidget {{
                    background-color: {content_color};
                    border: none;
                }}
                #sidebarWidget {{
                    background-color: {sidebar_color};
                    border: none;
                }}
                #appearancePage, #fontPage, #matlabPage, #logsPage, #aboutPage {{
                    background-color: {content_color};
                }}
                QLabel {{
                    background-color: transparent;
                }}
                QPushButton {{
                    background-color: transparent;
                }}
            """
        elif current_theme == "tokyo_night":
            # Tokyo Night theme
            button_style = f"""
                QPushButton {{
                    background-color: transparent;
                    border: 1px solid {accent_color};
                    border-radius: 5px;
                    padding: 4px 10px;
                    color: #a9b1d6;
                    min-height: 26px;
                }}
                QPushButton:hover {{
                    background-color: rgba(42, 195, 222, 0.1);
                }}
                QPushButton:pressed {{
                    background-color: rgba(42, 195, 222, 0.2);
                }}
            """
            
            combo_style = f"""
                QComboBox {{
                    background-color: {sidebar_color};
                    border: 1px solid #414868;
                    border-radius: 5px;
                    padding: 4px 4px 4px 15px;
                    color: #a9b1d6;
                    min-height: 26px;
                }}
                QComboBox:hover {{
                    border: 1px solid {accent_color};
                }}
                QComboBox QAbstractItemView {{
                    background-color: {sidebar_color};
                    color: #a9b1d6;
                    selection-background-color: rgba(42, 195, 222, 0.2);
                }}
            """
            
            label_style = """
                QLabel {
                    color: #a9b1d6;
                    font-size: 13.5px;
                    background-color: transparent;
                }
            """
            
            group_style = f"""
                QGroupBox {{
                    margin-top: 1.5em;
                    padding: 20px;
                    background-color: {content_color};
                    border: none;
                    border-radius: 8px;
                }}
                QGroupBox::title {{
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                    color: #a9b1d6;
                }}
            """
            
            text_edit_style = f"""
                QTextEdit {{
                    background-color: {sidebar_color};
                    border: 1px solid #414868;
                    border-radius: 5px;
                    color: #a9b1d6;
                }}
            """
            
            list_widget_style = f"""
                QListWidget {{
                    background-color: {sidebar_color};
                    border-right: none;
                    border-left: none;
                    border-top: none;
                    color: #a9b1d6;
                    padding: 10px 0px 0px 0px;
                }}
                QListWidget::item {{
                    padding: 6px 8px 6px 30px;
                    border-radius: 4px;
                    margin: 2px 8px 2px 8px;
                }}
                QListWidget::item:selected {{
                    background-color: {accent_color};
                    color: #16161e;
                    font-weight: bold;
                }}
                QListWidget::item:hover {{
                    background-color: rgba(42, 195, 222, 0.3);
                    border: 1px solid rgba(42, 195, 222, 0.5);
                }}
            """
            
            splitter_style = """
                QSplitter::handle {
                    background-color: #5a5f6b;
                    width: 1px;
                }
            """
            
            main_style = f"""
                QDialog {{
                    background-color: {content_color};
                }}
                QStackedWidget {{
                    background-color: {content_color};
                }}
                #contentWidget {{
                    background-color: {content_color};
                    border: none;
                }}
                #sidebarWidget {{
                    background-color: {sidebar_color};
                    border: none;
                }}
                #appearancePage, #fontPage, #matlabPage, #logsPage, #aboutPage {{
                    background-color: {content_color};
                }}
                QLabel {{
                    background-color: transparent;
                }}
                QPushButton {{
                    background-color: transparent;
                }}
                QScrollArea {{
                    background-color: {content_color};
                }}
                QWidget#appearanceGroup, QWidget#fontGroup, QWidget#matlabGroup, QWidget#logsGroup, QWidget#aboutGroup {{
                    background-color: {content_color};
                }}
                QWidget {{
                    background-color: {content_color};
                }}
            """
        elif current_theme == "aura":
            # Aura theme
            button_style = f"""
                QPushButton {{
                    background-color: transparent;
                    border: 1px solid {accent_color};
                    border-radius: 5px;
                    padding: 4px 10px;
                    color: #edecee;
                    min-height: 26px;
                }}
                QPushButton:hover {{
                    background-color: rgba(162, 119, 255, 0.1);
                }}
                QPushButton:pressed {{
                    background-color: rgba(162, 119, 255, 0.2);
                }}
            """
            
            combo_style = f"""
                QComboBox {{
                    background-color: {sidebar_color};
                    border: 1px solid #6d6d6d;
                    border-radius: 5px;
                    padding: 4px 4px 4px 15px;
                    color: #edecee;
                    min-height: 26px;
                }}
                QComboBox:hover {{
                    border: 1px solid {accent_color};
                }}
                QComboBox QAbstractItemView {{
                    background-color: {sidebar_color};
                    color: #edecee;
                    selection-background-color: rgba(162, 119, 255, 0.2);
                }}
            """
            
            label_style = """
                QLabel {
                    color: #edecee;
                    font-size: 13.5px;
                    background-color: transparent;
                }
            """
            
            group_style = f"""
                QGroupBox {{
                    margin-top: 1.5em;
                    padding: 20px;
                    background-color: {content_color};
                    border: none;
                    border-radius: 8px;
                }}
                QGroupBox::title {{
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                    color: #edecee;
                }}
            """
            
            text_edit_style = f"""
                QTextEdit {{
                    background-color: {sidebar_color};
                    border: 1px solid #6d6d6d;
                    border-radius: 5px;
                    color: #edecee;
                }}
            """
            
            list_widget_style = f"""
                QListWidget {{
                    background-color: {sidebar_color};
                    border-right: none;
                    border-left: none;
                    border-top: none;
                    color: #edecee;
                    padding: 10px 0px 0px 0px;
                }}
                QListWidget::item {{
                    padding: 6px 8px 6px 30px;
                    border-radius: 4px;
                    margin: 2px 8px 2px 8px;
                }}
                QListWidget::item:selected {{
                    background-color: {accent_color};
                    color: #12111a;
                    font-weight: bold;
                }}
                QListWidget::item:hover {{
                    background-color: rgba(162, 119, 255, 0.3);
                    border: 1px solid rgba(162, 119, 255, 0.5);
                }}
            """
            
            splitter_style = """
                QSplitter::handle {
                    background-color: #8d8d8d;
                    width: 1px;
                }
            """
            
            main_style = f"""
                QDialog {{
                    background-color: {content_color};
                }}
                QStackedWidget {{
                    background-color: {content_color};
                }}
                #contentWidget {{
                    background-color: {content_color};
                    border: none;
                }}
                #sidebarWidget {{
                    background-color: {sidebar_color};
                    border: none;
                }}
                #appearancePage, #fontPage, #matlabPage, #logsPage, #aboutPage {{
                    background-color: {content_color};
                }}
                QLabel {{
                    background-color: transparent;
                }}
                QPushButton {{
                    background-color: transparent;
                }}
                QScrollArea {{
                    background-color: {content_color};
                }}
                QWidget#appearanceGroup, QWidget#fontGroup, QWidget#matlabGroup, QWidget#logsGroup, QWidget#aboutGroup {{
                    background-color: {content_color};
                }}
                QWidget {{
                    background-color: {content_color};
                }}
            """
        elif current_theme == "anysphere":
            # Anysphere theme
            button_style = f"""
                QPushButton {{
                    background-color: transparent;
                    border: 1px solid {accent_color};
                    border-radius: 5px;
                    padding: 4px 10px;
                    color: #e6edf3;
                    min-height: 26px;
                }}
                QPushButton:hover {{
                    background-color: rgba(230, 200, 149, 0.1);
                }}
                QPushButton:pressed {{
                    background-color: rgba(230, 200, 149, 0.2);
                }}
            """
            
            combo_style = f"""
                QComboBox {{
                    background-color: {sidebar_color};
                    border: 1px solid #30363d;
                    border-radius: 5px;
                    padding: 4px 4px 4px 15px;
                    color: #e6edf3;
                    min-height: 26px;
                }}
                QComboBox:hover {{
                    border: 1px solid {accent_color};
                }}
                QComboBox QAbstractItemView {{
                    background-color: {content_color};
                    color: #e6edf3;
                    selection-background-color: rgba(230, 200, 149, 0.2);
                }}
            """
            
            label_style = """
                QLabel {
                    color: #e6edf3;
                    font-size: 13.5px;
                    background-color: transparent;
                }
            """
            
            group_style = f"""
                QGroupBox {{
                    margin-top: 1.5em;
                    padding: 20px;
                    background-color: {content_color};
                    border: none;
                    border-radius: 8px;
                }}
                QGroupBox::title {{
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                    color: #e6edf3;
                }}
            """
            
            text_edit_style = f"""
                QTextEdit {{
                    background-color: {content_color};
                    border: 1px solid #30363d;
                    border-radius: 5px;
                    color: #e6edf3;
                }}
            """
            
            list_widget_style = f"""
                QListWidget {{
                    background-color: {sidebar_color};
                    border-right: none;
                    border-left: none;
                    border-top: none;
                    color: #e6edf3;
                    padding: 10px 0px 0px 0px;
                }}
                QListWidget::item {{
                    padding: 6px 8px 6px 30px;
                    border-radius: 4px;
                    margin: 2px 8px 2px 8px;
                }}
                QListWidget::item:selected {{
                    background-color: {accent_color};
                    color: {sidebar_color};
                    font-weight: bold;
                }}
                QListWidget::item:hover {{
                    background-color: rgba(230, 200, 149, 0.3);
                    border: 1px solid rgba(230, 200, 149, 0.5);
                }}
            """
            
            splitter_style = """
                QSplitter::handle {
                    background-color: #4a4f57;
                    width: 1px;
                }
            """
            
            main_style = f"""
                QDialog {{
                    background-color: {content_color};
                }}
                QStackedWidget {{
                    background-color: {content_color};
                }}
                #contentWidget {{
                    background-color: {content_color};
                    border: none;
                }}
                #sidebarWidget {{
                    background-color: {sidebar_color};
                    border: none;
                }}
                #navList {{
                    background-color: {sidebar_color};
                }}
                #appearancePage, #fontPage, #matlabPage, #logsPage, #aboutPage {{
                    background-color: {content_color};
                }}
                QLabel {{
                    background-color: transparent;
                }}
                QPushButton {{
                    background-color: transparent;
                }}
                QScrollArea {{
                    background-color: {content_color};
                }}
                QWidget#appearanceGroup, QWidget#fontGroup, QWidget#matlabGroup, QWidget#logsGroup, QWidget#aboutGroup {{
                    background-color: {content_color};
                }}
                QWidget {{
                    background-color: {content_color};
                }}
            """
        else:  # Default fallback style
            # Generic dark theme style with the current accent color
            button_style = f"""
                QPushButton {{
                    background-color: transparent;
                    border: 1px solid {accent_color};
                    border-radius: 5px;
                    padding: 4px 10px;
                    color: #e6edf3;
                    min-height: 26px;
                }}
                QPushButton:hover {{
                    background-color: rgba({self._get_rgba_from_hex(accent_color, 0.1)});
                }}
                QPushButton:pressed {{
                    background-color: rgba({self._get_rgba_from_hex(accent_color, 0.2)});
                }}
            """
            
            combo_style = f"""
                QComboBox {{
                    background-color: {sidebar_color};
                    border: 1px solid #30363d;
                    border-radius: 5px;
                    padding: 4px 4px 4px 15px;
                    color: #e6edf3;
                    min-height: 26px;
                }}
                QComboBox:hover {{
                    border: 1px solid {accent_color};
                }}
                QComboBox QAbstractItemView {{
                    background-color: {content_color};
                    color: #e6edf3;
                    selection-background-color: rgba({self._get_rgba_from_hex(accent_color, 0.2)});
                }}
            """
            
            label_style = """
                QLabel {
                    color: #e6edf3;
                    font-size: 13.5px;
                    background-color: transparent;
                }
            """
            
            group_style = f"""
                QGroupBox {{
                    margin-top: 1.5em;
                    padding: 20px;
                    background-color: {content_color};
                    border: none;
                    border-radius: 8px;
                }}
                QGroupBox::title {{
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                    color: #e6edf3;
                }}
            """
            
            text_edit_style = f"""
                QTextEdit {{
                    background-color: {content_color};
                    border: 1px solid #30363d;
                    border-radius: 5px;
                    color: #e6edf3;
                }}
            """
            
            list_widget_style = f"""
                QListWidget {{
                    background-color: {sidebar_color};
                    border-right: none;
                    border-left: none;
                    border-top: none;
                    color: #e6edf3;
                    padding: 10px 0px 0px 0px;
                }}
                QListWidget::item {{
                    padding: 6px 8px 6px 30px;
                    border-radius: 4px;
                    margin: 2px 8px 2px 8px;
                }}
                QListWidget::item:selected {{
                    background-color: {accent_color};
                    color: {sidebar_color};
                    font-weight: bold;
                }}
                QListWidget::item:hover {{
                    background-color: rgba({self._get_rgba_from_hex(accent_color, 0.3)});
                    border: 1px solid rgba({self._get_rgba_from_hex(accent_color, 0.5)});
                }}
            """
            
            splitter_style = """
                QSplitter::handle {
                    background-color: #4a4f57;
                    width: 1px;
                }
            """
            
            main_style = f"""
                QDialog {{
                    background-color: {content_color};
                }}
                QStackedWidget {{
                    background-color: {content_color};
                }}
                #contentWidget {{
                    background-color: {content_color};
                    border: none;
                }}
                #sidebarWidget {{
                    background-color: {sidebar_color};
                    border: none;
                }}
                #navList {{
                    background-color: {sidebar_color};
                }}
                #appearancePage, #fontPage, #matlabPage, #logsPage, #aboutPage {{
                    background-color: {content_color};
                }}
                QLabel {{
                    background-color: transparent;
                }}
                QPushButton {{
                    background-color: transparent;
                }}
                QScrollArea {{
                    background-color: {content_color};
                }}
                QWidget#appearanceGroup, QWidget#fontGroup, QWidget#matlabGroup, QWidget#logsGroup, QWidget#aboutGroup {{
                    background-color: {content_color};
                }}
                QWidget {{
                    background-color: {content_color};
                }}
            """
        
        # Apply styles to UI components
        self.setStyleSheet(main_style)
        
        for button in self.findChildren(QPushButton):
            button.setStyleSheet(button_style)
            button.setMinimumHeight(26)
        
        for combo in self.findChildren(QComboBox):
            combo.setStyleSheet(combo_style)
            combo.setMinimumHeight(26)
        
        for label in self.findChildren(QLabel):
            label.setStyleSheet(label_style)
        
        for group in self.findChildren(QGroupBox):
            group.setStyleSheet(group_style)
        
        for text_edit in self.findChildren(QTextEdit):
            text_edit.setStyleSheet(text_edit_style)
        
        for list_widget in self.findChildren(QListWidget):
            list_widget.setStyleSheet(list_widget_style)
        
        for splitter in self.findChildren(QSplitter):
            splitter.setStyleSheet(splitter_style)
            
        # Get individual widgets by object name and apply specific styles
        sidebar_widget = self.findChild(QWidget, "sidebarWidget")
        if sidebar_widget:
            sidebar_widget.setStyleSheet(f"background-color: {sidebar_color};")
            
        content_widget = self.findChild(QWidget, "contentWidget")
        if content_widget:
            content_widget.setStyleSheet(f"background-color: {content_color};")
            
        nav_list = self.findChild(QListWidget, "navList")
        if nav_list:
            nav_list.setStyleSheet(list_widget_style)
            
        # Apply to stacked pages
        for page_name in ["appearancePage", "fontPage", "matlabPage", "logsPage", "aboutPage"]:
            page = self.findChild(QWidget, page_name)
            if page:
                page.setStyleSheet(f"background-color: {content_color};")
        
        # Update font preview if it exists
        if hasattr(self, 'preview_text'):
            self.update_font_preview()
        
        # Ensure consistent sizing
        self.setFixedSize(self.WINDOW_WIDTH, self.WINDOW_HEIGHT)

    def on_theme_changed(self, theme_name):
        """Handle theme change event"""
        # Convert display names to internal theme names
        theme_map = {
            "Tokyo night": "tokyo_night",
            "Aura": "aura",
            "Light": "light",
            "Anysphere": "anysphere"
        }
        
        theme_key = theme_map.get(theme_name, "anysphere")
        
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
                return settings.get('theme', 'anysphere')  # Return 'anysphere' if theme not found
        except FileNotFoundError:
            # Create default settings file if it doesn't exist
            default_settings = {
                "theme": "anysphere",
                "input_mode": "LaTeX",
                "angle_mode": "Radian"
            }
            with open('settings.json', 'w') as f:
                json.dump(default_settings, f, indent=4)
            return "anysphere"
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            return "anysphere"
        
    def _create_matlab_credentials(self):
        """Create the MATLAB credentials group"""
        group_box = QGroupBox()
        group_box.setObjectName("matlabGroup")
        
        current_theme = getattr(self.parent, 'current_theme', 'anysphere')
        if current_theme == "light":
            credentials_style = """
                QGroupBox {
                    padding: 15px 20px 20px 20px;
                    background-color: #f0f0f0;
                    border: none;
                    border-radius: 8px;
                }
                QLabel {
                    color: #333333;
                }
            """
        else:
            credentials_style = """
                QGroupBox {
                    padding: 15px 20px 20px 20px;
                    background-color: #1A1D21;
                    border: none;
                    border-radius: 8px;
                }
                QLabel {
                    color: #e6edf3;
                }
            """
        
        group_box.setStyleSheet(credentials_style)
        
        main_layout = QVBoxLayout()
        
        # Title directly in main layout
        title_label = QLabel("MATLAB License")
        title_label.setFont(QFont(self.DISPLAY_FONT, 14, QFont.Bold))
        main_layout.addWidget(title_label)
        
        # Add spacing after title
        main_layout.addSpacing(10)
        
        credentials_layout = QVBoxLayout()
        credentials_layout.setSpacing(20)
        
        license_description = QLabel(
            "A valid MATLAB license is required for MATLAB engine computations. "
            "Without a valid license, the calculator will operate in SymPy-only "
            "mode with limited functionality."
        )
        license_description.setFont(QFont(self.DISPLAY_FONT, 12.5))
        license_description.setWordWrap(True)
        license_description.setStyleSheet("margin-bottom: 20px;")
        credentials_layout.addWidget(license_description)
        
        credentials_layout.addSpacing(10)
        
        self.license_status_label = QLabel("License Status: Not Checked")
        self.license_status_label.setFont(QFont(self.DISPLAY_FONT, 13.5))
        self.license_status_label.setAlignment(Qt.AlignLeft)
        
        credentials_layout.addWidget(self.license_status_label)
        
        self.check_license_button = QPushButton("Check License")
        self.check_license_button.setFont(QFont(self.DISPLAY_FONT, 13.5))
        self.check_license_button.setMinimumHeight(26)
        self.check_license_button.clicked.connect(self.check_matlab_license)
        
        credentials_layout.addWidget(self.check_license_button)
        credentials_layout.addStretch()
        main_layout.addLayout(credentials_layout)
        group_box.setLayout(main_layout)
        
        return group_box

    def _create_about_group(self):
        """Create the about group"""
        group_box = QGroupBox()
        group_box.setObjectName("aboutGroup")
        
        # Set style based on current theme
        current_theme = getattr(self.parent, 'current_theme', 'anysphere')
        if current_theme == "light":
            about_style = """
                QGroupBox {
                    padding: 15px 20px 20px 20px;
                    background-color: #f0f0f0;
                    border: none;
                    border-radius: 8px;
                }
            """
            # Simple text without HTML styling
            about_text = """
            <h2>VorTeX Calculator</h2>
            <p>Version 1.0.3</p>
            <p>A LaTeX-based calculator with MATLAB/Sympy integration.</p>
            <p>© 2025 VorTeX Team</p>
            """
        else:  # anysphere/dark theme
            about_style = """
                QGroupBox {
                    padding: 15px 20px 20px 20px;
                    background-color: #1A1D21;
                    border: none;
                    border-radius: 8px;
                }
            """
            # Simple text without HTML styling
            about_text = """
            <h2>VorTeX Calculator</h2>
            <p>Version 1.0.3</p>
            <p>A LaTeX-based calculator with MATLAB/Sympy integration.</p>
            <p>© 2025 VorTeX Team</p>
            """
        
        group_box.setStyleSheet(about_style)
        
        main_layout = QVBoxLayout()
        
        # Title directly in main layout
        title_label = QLabel("About")
        title_label.setFont(QFont(self.DISPLAY_FONT, 14, QFont.Bold))
        if current_theme == "light":
            title_label.setStyleSheet("color: #000000;")
        else:
            title_label.setStyleSheet("color: #ffffff;")
        main_layout.addWidget(title_label)
        
        # Add spacing after title
        main_layout.addSpacing(10)
        
        about_layout = QVBoxLayout()
        about_layout.setSpacing(10)
        
        about_label = QLabel(about_text)
        about_label.setFont(QFont(self.DISPLAY_FONT, 12))
        about_label.setTextFormat(Qt.RichText)
        about_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        about_label.setOpenExternalLinks(True)
        
        # Explicitly set the text color based on theme
        if current_theme == "light":
            about_label.setStyleSheet("color: #000000;")
        else:
            about_label.setStyleSheet("color: #ffffff;")
        
        about_layout.addWidget(about_label)
        
        main_layout.addLayout(about_layout)
        group_box.setLayout(main_layout)
        
        return group_box

    def _create_logs_group(self):
        """Create the debug logs group"""
        group_box = QGroupBox()
        group_box.setObjectName("logsGroup")
        
        # Set style based on current theme
        current_theme = getattr(self.parent, 'current_theme', 'anysphere')
        if current_theme == "light":
            logs_style = """
                QGroupBox {
                    padding: 15px 20px 20px 20px;
                    background-color: #f0f0f0;
                    border: none;
                    border-radius: 8px;
                }
                QLabel {
                    color: #333333;
                }
            """
        else:  # anysphere/dark theme
            logs_style = """
                QGroupBox {
                    padding: 15px 20px 20px 20px;
                    background-color: #1A1D21;
                    border: none;
                    border-radius: 8px;
                }
                QLabel {
                    color: #e6edf3;
                }
            """
        
        group_box.setStyleSheet(logs_style)
        
        main_layout = QVBoxLayout()
        
        # Title directly in main layout
        title_label = QLabel("Debug Logs")
        title_label.setFont(QFont(self.DISPLAY_FONT, 14, QFont.Bold))
        main_layout.addWidget(title_label)
        
        # Add spacing after title
        main_layout.addSpacing(10)
        
        logs_layout = QVBoxLayout()
        logs_layout.setSpacing(10)
        
        # Add log view window
        try:
            log_file_path = 'calculator.log'
            log_content = ""
            if os.path.exists(log_file_path):
                with open(log_file_path, 'r') as f:
                    log_content = f.read()
            
            self.log_display = QTextEdit()
            self.log_display.setReadOnly(True)
            self.log_display.setFont(QFont(self.FONT_FAMILY, 12))
            self.log_display.setText(log_content)
            self.log_display.setMinimumHeight(300)  # Reduced height to make room for buttons
            
            logs_layout.addWidget(self.log_display)
        except Exception as e:
            logger.error(f"Error loading logs: {e}")
            error_label = QLabel(f"Error loading logs: {str(e)}")
            error_label.setFont(QFont(self.DISPLAY_FONT, 13.5))  # Decreased font size to 13.5
            logs_layout.addWidget(error_label)
        
        # Button layout at the bottom with fixed position
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        button_layout.setContentsMargins(0, 10, 0, 0)  # Add top margin to separate from log display
        
        # Clear logs button
        self.clear_logs_button = QPushButton("Clear Logs")
        self.clear_logs_button.setFont(QFont(self.DISPLAY_FONT, 13.5))  # Decreased font size to 13.5
        self.clear_logs_button.setMinimumHeight(26)  # Reduced height to 26
        self.clear_logs_button.clicked.connect(self.clear_logs)
        
        # Export logs button
        self.export_logs_button = QPushButton("Export Logs")
        self.export_logs_button.setFont(QFont(self.DISPLAY_FONT, 13.5))  # Decreased font size to 13.5
        self.export_logs_button.setMinimumHeight(26)  # Reduced height to 26
        self.export_logs_button.clicked.connect(self.export_logs)
        
        button_layout.addWidget(self.clear_logs_button)
        button_layout.addWidget(self.export_logs_button)
        button_layout.addStretch()
        
        logs_layout.addLayout(button_layout)
        main_layout.addLayout(logs_layout)
        group_box.setLayout(main_layout)
        
        return group_box

    def show_logs(self):
        """Open a dialog to show the log file contents"""
        try:
            log_file_path = 'calculator.log'
            with open(log_file_path, 'r') as f:
                log_content = f.read()
            
            dialog = QDialog(self)
            dialog.setWindowTitle("Debug Logs")
            dialog.setFixedSize(700, 500)
            
            layout = QVBoxLayout()
            
            log_display = QTextEdit()
            log_display.setReadOnly(True)
            log_display.setFont(QFont(self.DISPLAY_FONT, 12))
            log_display.setText(log_content)
            
            layout.addWidget(log_display)
            dialog.setLayout(layout)
            dialog.exec_()
            
        except FileNotFoundError:
            QMessageBox.information(self, "Info", "No log file found.")
        except Exception as e:
            logger.error(f"Error showing logs: {e}")
            QMessageBox.warning(self, "Error", f"Failed to read logs: {str(e)}")

    def clear_logs(self):
        """Clear the application logs"""
        try:
            log_file = 'calculator.log'
            if os.path.exists(log_file):
                with open(log_file, 'w') as f:
                    f.write('')
                QMessageBox.information(self, "Logs Cleared", "Application logs have been cleared successfully.")
                
                # Update the log display if it exists
                if hasattr(self, 'log_display'):
                    self.log_display.clear()
            else:
                QMessageBox.information(self, "No Logs", "No log file found.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to clear logs: {str(e)}")

    def check_license_validity(self):
        """Check MATLAB License validity by sending a request to the MATLAB server"""
        try:
            eng = matlab.engine.start_matlab()
            eng.quit()
            self.license_status_label.setText("License Status: Valid")
        except Exception as e:
            self.license_status_label.setText("License Status: Invalid")
            logger.error(f"Error checking MATLAB License: {e}")

    def _create_font_settings_group(self):
        """Create the font settings group"""
        group_box = QGroupBox()
        group_box.setObjectName("fontGroup")
        
        current_theme = getattr(self.parent, 'current_theme', 'anysphere')
        if current_theme == "light":
            font_style = """
                QGroupBox {
                    margin-top: 1.5em;
                    padding: 15px 20px 20px 20px;
                    background-color: #f0f0f0;
                    border: none;
                    border-radius: 8px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                    color: #333333;
                }
                QLabel {
                    color: #333333;
                }
            """
        else:
            font_style = """
                QGroupBox {
                    margin-top: 1.5em;
                    padding: 15px 20px 20px 20px;
                    background-color: #1A1D21;
                    border: none;
                    border-radius: 8px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                    color: #e6edf3;
                }
                QLabel {
                    color: #e6edf3;
                }
            """
        
        group_box.setStyleSheet(font_style)
        
        font_layout = QVBoxLayout()
        
        # Title directly in main layout
        title_label = QLabel("Font Settings")
        title_label.setFont(QFont(self.DISPLAY_FONT, 14, QFont.Bold))
        font_layout.addWidget(title_label)
        
        # Add spacing after title
        font_layout.addSpacing(10)
        
        # Font description
        font_description = QLabel(
            "Customize the fonts used throughout the calculator interface. "
            "Font family affects readability of mathematical expressions. "
        )
        font_description.setFont(QFont(self.DISPLAY_FONT, 12.5))
        font_description.setWordWrap(True)
        font_description.setStyleSheet("margin-bottom: 20px;")
        font_layout.addWidget(font_description)
        
        font_layout.addSpacing(10)
        
        font_family_layout = QHBoxLayout()
        font_family_label = QLabel("Math Input Font:")
        font_family_label.setFont(QFont(self.DISPLAY_FONT, 13.5))
        
        self.font_family_combo = QComboBox()
        self.font_family_combo.setFont(QFont(self.FONT_FAMILY, 12))
        self.font_family_combo.addItems(["Menlo", "Monaco", "Courier New", "Consolas", "Monaspace Neon"])
        self.font_family_combo.setCurrentText(self.FONT_FAMILY)
        self.font_family_combo.setMinimumHeight(26)
        self.font_family_combo.setMinimumWidth(250)
        self.font_family_combo.currentTextChanged.connect(self.update_font_preview)
        
        font_family_layout.addWidget(font_family_label)
        font_family_layout.addWidget(self.font_family_combo)
        font_family_layout.addStretch()
        
        font_size_layout = QHBoxLayout()
        font_size_label = QLabel("Font Size:")
        font_size_label.setFont(QFont(self.DISPLAY_FONT, 13.5))
        
        self.font_size_combo = QComboBox()
        self.font_size_combo.setFont(QFont(self.FONT_FAMILY, 12))
        self.font_size_combo.addItems(["10", "11", "12", "13", "14", "16", "18"])
        self.font_size_combo.setCurrentText(str(self.FONT_SIZE))
        self.font_size_combo.setMinimumHeight(26)
        self.font_size_combo.setMinimumWidth(100)
        self.font_size_combo.currentTextChanged.connect(self.update_font_preview)
        
        font_size_layout.addWidget(font_size_label)
        font_size_layout.addWidget(self.font_size_combo)
        font_size_layout.addStretch()
        
        preview_label = QLabel("Math Input Preview:")
        preview_label.setFont(QFont(self.DISPLAY_FONT, 13.5))
        
        self.preview_text = QTextEdit()
        self.preview_text.setFont(QFont(self.FONT_FAMILY, int(self.FONT_SIZE)))
        self.preview_text.setPlainText("\\int_{0}^{\\pi} \\sin(x) dx = 2\n\\frac{d}{dx}[x^2] = 2x\n\\sum_{i=1}^{n} i = \\frac{n(n+1)}{2}")
        self.preview_text.setFixedHeight(120)
        
        self.apply_font_button = QPushButton("Apply Font Settings")
        self.apply_font_button.setFont(QFont(self.DISPLAY_FONT, 13.5))
        self.apply_font_button.setMinimumHeight(26)
        self.apply_font_button.clicked.connect(self.apply_font_settings)
        
        font_layout.addLayout(font_family_layout)
        font_layout.addLayout(font_size_layout)
        font_layout.addWidget(preview_label)
        font_layout.addWidget(self.preview_text)
        font_layout.addWidget(self.apply_font_button)
        
        group_box.setLayout(font_layout)
        
        return group_box

    def on_font_family_changed(self, font_family):
        """Handle font family change event"""
        if self.parent:
            # Check if the parent has a set_font_family method
            if hasattr(self.parent, 'set_font_family'):
                self.parent.set_font_family(font_family)
            
            # Update the preview
            self.update_font_preview()
            
            # Save the setting
            self.save_font_settings({'family': font_family})
            
            # Apply theme to update UI
            self.apply_theme()

    def on_font_size_changed(self, font_size):
        """Handle font size change event"""
        if self.parent:
            # Check if the parent has a set_font_size method
            if hasattr(self.parent, 'set_font_size'):
                self.parent.set_font_size(int(font_size))
            
            # Update the preview
            self.update_font_preview()
            
            # Save the setting
            self.save_font_settings({'size': int(font_size)})
            
            # Apply theme to update UI
            self.apply_theme()

    def update_font_preview(self):
        """Update the font preview based on the current font settings"""
        if hasattr(self, 'font_family_combo') and hasattr(self, 'font_size_combo'):
            font_family = self.font_family_combo.currentText()
            font_size = self.font_size_combo.currentText()
            self.preview_text.setFont(QFont(font_family, int(font_size)))
            
            # Preserve the style while updating the font
            current_theme = getattr(self.parent, 'current_theme', 'anysphere')
            if current_theme == "light":
                preview_style = """
                    background-color: #ffffff;
                    border: 1px solid #c0c0c0;
                    border-radius: 5px;
                    color: #333333;
                    font-family: "%s";
                    font-size: %spt;
                """ % (font_family, font_size)
            else:
                preview_style = """
                    background-color: #1A1D21;
                    border: 1px solid #30363d;
                    border-radius: 5px;
                    color: #e6edf3;
                    font-family: "%s";
                    font-size: %spt;
                """ % (font_family, font_size)
            
            self.preview_text.setStyleSheet(preview_style)
            self.preview_text.setText("\\int_{0}^{\\pi} \\sin(x) dx = 2\n\\frac{d}{dx}[x^2] = 2x\n\\sum_{i=1}^{n} i = \\frac{n(n+1)}{2}")

    def save_font_settings(self, settings):
        """Save font settings to settings.json"""
        try:
            # Load existing settings
            try:
                with open('settings.json', 'r') as f:
                    existing_settings = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                # Create default settings if file doesn't exist or is invalid
                existing_settings = {
                    "theme": "anysphere",
                    "input_mode": "LaTeX",
                    "angle_mode": "Degree",
                    "font_settings": {
                        "family": self.FONT_FAMILY,
                        "size": self.FONT_SIZE
                    }
                }
            
            # Ensure font_settings exists in the settings
            if 'font_settings' not in existing_settings:
                existing_settings['font_settings'] = {
                    "family": self.FONT_FAMILY,
                    "size": self.FONT_SIZE
                }
            
            # Update font settings with new values
            for key, value in settings.items():
                existing_settings['font_settings'][key] = value
            
            # Write updated settings to file
            with open('settings.json', 'w') as f:
                json.dump(existing_settings, f, indent=4)
                
            logger.info(f"Font settings saved: {existing_settings['font_settings']}")
                
        except Exception as e:
            logger.error(f"Error saving font settings: {e}")

    def load_font_settings(self):
        """Load font settings from settings.json"""
        try:
            with open('settings.json', 'r') as f:
                settings = json.load(f)
                # If no font settings exist, use the detected default
                if 'font_settings' not in settings:
                    settings['font_settings'] = {
                        "family": self.FONT_FAMILY,
                        "size": self.FONT_SIZE
                    }
                    # Save the updated settings
                    with open('settings.json', 'w') as f_write:
                        json.dump(settings, f_write, indent=4)
                return settings.get('font_settings', {})
        except FileNotFoundError:
            # Create default font settings file if it doesn't exist
            default_settings = {
                "theme": "anysphere",
                "input_mode": "LaTeX",
                "angle_mode": "Degree",
                "font_settings": {
                    "family": self.FONT_FAMILY,
                    "size": self.FONT_SIZE
                }
            }
            with open('settings.json', 'w') as f:
                json.dump(default_settings, f, indent=4)
            return default_settings.get('font_settings', {})
        except Exception as e:
            logger.error(f"Error loading font settings: {e}")
            return {"family": self.FONT_FAMILY, "size": self.FONT_SIZE}

    def apply_font_settings(self):
        """Apply the selected font settings"""
        font_family = self.font_family_combo.currentText()
        font_size = int(self.font_size_combo.currentText())
        
        # Save settings
        settings = self.load_font_settings()
        settings['family'] = font_family
        settings['size'] = font_size
        self.save_font_settings(settings)
        
        # Apply to parent if available
        if hasattr(self.parent, 'set_font_family'):
            self.parent.set_font_family(font_family)
        if hasattr(self.parent, 'set_font_size'):
            self.parent.set_font_size(font_size)
        
        # Update FONT_FAMILY and FONT_SIZE class variables
        self.FONT_FAMILY = font_family
        self.FONT_SIZE = font_size
        
        # Show confirmation message
        QMessageBox.information(self, "Font Settings", "Font settings applied successfully!")

    def export_logs(self):
        """Export the application logs to a file"""
        try:
            log_file = 'calculator.log'
            if os.path.exists(log_file):
                file_dialog = QFileDialog()
                save_path, _ = file_dialog.getSaveFileName(
                    self, "Export Logs", os.path.expanduser("~/Desktop/vortex_logs.txt"), "Text Files (*.txt)"
                )
                
                if save_path:
                    with open(log_file, 'r') as src, open(save_path, 'w') as dst:
                        dst.write(src.read())
                    QMessageBox.information(self, "Logs Exported", f"Application logs have been exported to {save_path}")
            else:
                QMessageBox.information(self, "No Logs", "No log file found to export.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to export logs: {str(e)}")

    def check_matlab_license(self):
        """Check the MATLAB license status"""
        try:
            # This is a placeholder - in a real app, you would check the actual license
            import matlab.engine
            self.license_status_label.setText("License Status: Valid")
            QMessageBox.information(self, "License Status", "MATLAB license is valid and active.")
        except Exception as e:
            self.license_status_label.setText("License Status: Invalid")
            QMessageBox.warning(self, "License Status", f"MATLAB license check failed: {str(e)}")

    def _get_rgba_from_hex(self, hex_color, alpha):
        """Helper method to convert a hex color to an RGBA tuple"""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        else:
            raise ValueError("Invalid hex color format")
        return f"{r}, {g}, {b}, {int(255 * alpha)}"
