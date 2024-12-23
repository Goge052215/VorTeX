from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPalette, QColor

class ThemeManager:
    def __init__(self):
        self.themes = {
            "tokyo_night": get_tokyo_night_theme(),
            "aura": get_aura_theme(),
            "light": get_light_theme()
        }

    def apply_theme(self, widget: QWidget, theme_name: str):
        if theme_name not in self.themes:
            return
        
        theme = self.themes[theme_name]
        widget.setStyleSheet(theme["main_widget"])
        return theme

def get_tokyo_night_theme():
    return {
        "main_widget": """
            QWidget {
                background-color: #1a1b26;
                color: #a9b1d6;
            }
            QTextEdit, QComboBox {
                background-color: #24283b;
                color: #a9b1d6;
                border: 1px solid #414868;
                border-radius: 5px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(path/to/down_arrow.png);
                width: 12px;
                height: 12px;
            }
            QPushButton {
                background-color: #3d59a1;
                color: #ffffff;
                border: none;
                padding: 8px 15px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2ac3de;
            }
            QLabel {
                color: #a9b1d6;
            }
        """,
        "theme_button": """
            QPushButton {
                background-color: #3d59a1;
                color: #ffffff;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2ac3de;
            }
        """,
        "text_color": "#a9b1d6",
        "border_color": "#414868",
        "title_color": "#ffffff",
        "colors": {
            "background": "#1a1b26",
            "text": "#a9b1d6",
            "button": "#3d59a1",
            "button_text": "#ffffff",
            "button_hover": "#2ac3de",
            "input_bg": "#24283b",
            "border": "#414868",
            "title": "#ffffff"
        }
    }

def get_aura_theme():
    return {
        "main_widget": """
            QWidget {
                background-color: #15141b;
                color: #edecee;
            }
            QTextEdit, QComboBox {
                background-color: #1c1b22;
                color: #edecee;
                border: 1px solid #6d6d6d;
                border-radius: 5px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(path/to/down_arrow.png);
                width: 12px;
                height: 12px;
            }
            QPushButton {
                background-color: #a277ff;
                color: #15141b;
                border: none;
                padding: 8px 15px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #61ffca;
            }
            QLabel {
                color: #edecee;
            }
        """,
        "theme_button": """
            QPushButton {
                background-color: #a277ff;
                color: #15141b;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #61ffca;
            }
        """,
        "text_color": "#edecee",
        "border_color": "#6d6d6d",
        "title_color": "#edecee",
        "colors": {
            "background": "#15141b",
            "text": "#edecee",
            "button": "#a277ff",
            "button_text": "#15141b",
            "button_hover": "#61ffca",
            "input_bg": "#1c1b22",
            "border": "#6d6d6d",
            "title": "#edecee"
        }
    }

def get_light_theme():
    return {
        "main_widget": """
            QWidget {
                background-color: #f0f0f0;
                color: #333333;
            }
            QTextEdit, QComboBox {
                background-color: #ffffff;
                color: #333333;
                border: 1px solid #cccccc;
                border-radius: 5px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(path/to/down_arrow_dark.png);
                width: 12px;
                height: 12px;
            }
            QPushButton {
                background-color: #ffffff;
                color: #4a90e2;
                border: 1px solid #4a90e2;
                padding: 8px 15px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e6f0ff;
            }
            QLabel {
                color: #333333;
            }
        """,
        "theme_button": """
            QPushButton {
                background-color: #4a90e2;
                color: #ffffff;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
        """,
        "text_color": "#333333",
        "border_color": "#cccccc",
        "title_color": "#333333",
        "colors": {
            "background": "#f0f0f0",
            "text": "#333333",
            "button": "#4a90e2",
            "button_text": "#ffffff",
            "button_hover": "#357abd",
            "input_bg": "#ffffff",
            "border": "#cccccc",
            "title": "#333333"
        }
    }
