from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPalette, QColor

class ThemeManager:
    def __init__(self):
        self.themes = {
            "tokyo_night": get_tokyo_night_theme(),
            "aura": get_aura_theme(),
            "light": get_light_theme(),
            "anysphere": get_anysphere_theme()
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
                border: none;
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
                background-color: transparent;
                color: #ffffff;
                border: 1px solid #3d59a1;
                padding: 8px 15px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(61, 89, 161, 0.1);
                border: 1px solid #2ac3de;
            }
            QLabel {
                color: #a9b1d6;
            }
        """,
        "theme_button": """
            QPushButton {
                background-color: transparent;
                color: #ffffff;
                border: 1px solid #3d59a1;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: rgba(42, 195, 222, 0.1);
                border: 1px solid #2ac3de;
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
                border: none;
            }
            QTextEdit, QComboBox {
                background-color: #15141b;
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
                background-color: transparent;
                color: #edecee;
                border: 1px solid #a277ff;
                padding: 8px 15px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(162, 119, 255, 0.1);
                border: 1px solid #61ffca;
            }
            QLabel {
                color: #edecee;
            }
        """,
        "theme_button": """
            QPushButton {
                background-color: transparent;
                color: #edecee;
                border: 1px solid #a277ff;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: rgba(97, 255, 202, 0.1);
                border: 1px solid #61ffca;
            }
        """,
        "text_color": "#edecee",
        "border_color": "#6d6d6d",
        "title_color": "#edecee",
        "colors": {
            "background": "#15141b",
            "text": "#edecee",
            "button": "#a277ff",
            "button_text": "#edecee",
            "button_hover": "#61ffca",
            "input_bg": "#15141b",
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
                border: none;
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
                background-color: transparent;
                color: #4a90e2;
                border: 1px solid #4a90e2;
                padding: 8px 15px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(74, 144, 226, 0.1);
                border: 1px solid #357abd;
            }
            QLabel {
                color: #333333;
            }
        """,
        "theme_button": """
            QPushButton {
                background-color: transparent;
                color: #4a90e2;
                border: 1px solid #4a90e2;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: rgba(53, 122, 189, 0.1);
                border: 1px solid #357abd;
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

def get_anysphere_theme():
    return {
        "main_widget": """
            QWidget {
                background-color: #0e1116;
                color: #e6edf3;
                border: none;
            }
            QTextEdit, QComboBox {
                background-color: #0e1116;
                color: #e6edf3;
                border: 1px solid #30363d;
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
                background-color: transparent;
                color: #ffffff;
                border: 1px solid #E6C895;
                padding: 8px 15px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(230, 200, 149, 0.1);
                border: 1px solid #E6C895;
            }
            QLabel {
                color: #e6edf3;
            }
        """,
        "theme_button": """
            QPushButton {
                background-color: transparent;
                color: #ffffff;
                border: 1px solid #E6C895;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: rgba(230, 200, 149, 0.1);
                border: 1px solid #E6C895;
            }
        """,
        "text_color": "#e6edf3",
        "border_color": "#30363d",
        "title_color": "#ffffff",
        "colors": {
            "background": "#0e1116",
            "text": "#e6edf3",
            "button": "#E6C895",
            "button_text": "#ffffff",
            "button_hover": "#E6C895",
            "input_bg": "#0e1116",
            "border": "#30363d",
            "title": "#ffffff"
        }
    }
