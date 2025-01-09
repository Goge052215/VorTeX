"""
Module for importing necessary Python packages and modules.
Handles installation of missing packages and imports required modules.
"""

import os
import sys
from typing import NoReturn


class PackageImporter:
    """Class for managing package imports and installations."""

    @staticmethod
    def import_pyqt5() -> NoReturn:
        """Install and import PyQt5 and related packages."""
        try:
            print("Installing PyQt5 and related packages...")
            packages = ["PyQt5", "sympy", "numpy"]
            for package in packages:
                os.system(f"{sys.executable} -m pip install {package}")

            # PyQt5 imports
            from PyQt5.QtWidgets import (
                QApplication, QWidget, QVBoxLayout, QLabel, QPushButton,
                QComboBox, QMessageBox, QTextEdit, QHBoxLayout, QGridLayout,
                QScrollArea, QSpacerItem, QSizePolicy, QMenu, QInputDialog
            )
            from PyQt5.QtGui import QFont, QIcon
            from PyQt5.QtCore import Qt

            # Additional package imports
            import sympy as sy
            import numpy as np

        except Exception as e:
            print(f"Error installing/importing PyQt5 packages: {e}")

    @staticmethod
    def import_matlab() -> NoReturn:
        """Install and import MATLAB-related packages."""
        try:
            print("Installing MATLAB packages...")
            packages = ["matlab", "matlabengine"]
            for package in packages:
                os.system(f"{sys.executable} -m pip install {package}")

            import matlab.engine

        except Exception as e:
            print(f"Error installing/importing MATLAB packages: {e}")

    @staticmethod
    def import_sympy() -> NoReturn:
        """Install and import SymPy-related packages."""
        try:
            print("Installing SymPy and related packages...")
            packages = ["sympy", "numpy", "latex2sympy2"]
            for package in packages:
                os.system(f"{sys.executable} -m pip install {package}")

            import sympy as sy
            from sympy import sympify, Symbol
            import numpy as np
            from latex2sympy2 import latex2sympy

        except Exception as e:
            print(f"Error installing/importing SymPy packages: {e}")

    @staticmethod
    def import_manim() -> NoReturn:
        """Install and import Manim-related packages."""
        try:
            print("Installing Manim and related packages...")
            # First install core dependencies
            core_packages = [
                "importlib_metadata",
                "pycairo",
                "pangocairo",
                "manimpango",
                "numpy",
                "pydub",
                "ffmpeg-python",
                "opencv-python"
            ]
            
            for package in core_packages:
                print(f"Installing {package}...")
                os.system(f"{sys.executable} -m pip install {package}")

            # Then install manim
            print("Installing manim...")
            os.system(f"{sys.executable} -m pip install manim")

            # Verify installations
            import importlib_metadata
            import manim
            print("Manim installation successful!")

        except Exception as e:
            print(f"Error installing/importing Manim packages: {e}")
            print("\nPlease try installing manually:")
            print("1. pip install importlib_metadata")
            print("2. pip install manim")
            print("3. For macOS: brew install cairo pango ffmpeg")
            print("4. For Windows: Check Manim installation guide")
