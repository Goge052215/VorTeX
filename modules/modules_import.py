"""
Module for importing necessary Python packages and modules.
Handles installation of missing packages and imports required modules.
"""

import os
import sys
from typing import NoReturn


class PackageImporter:
    @staticmethod
    def import_pyqt5() -> NoReturn:
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

            import sympy as sy
            import numpy as np

        except Exception as e:
            print(f"Error installing/importing PyQt5 packages: {e}")

    @staticmethod
    def import_matlab() -> NoReturn:
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
        try:
            print("Installing Manim and related packages...")
            
            # First try to use the built-in importlib.metadata
            try:
                from importlib import metadata as importlib_metadata
                print("Using built-in importlib.metadata")
            except ImportError:
                print("Installing importlib_metadata package...")
                os.system(f"{sys.executable} -m pip install importlib_metadata")
            
            core_packages = [
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

            print("Installing manim...")
            os.system(f"{sys.executable} -m pip install manim")

            # Verify installations
            import manim
            print("Manim installation successful!")
            
        except Exception as e:
            print(f"Error during Manim installation: {e}")
            print("\nTry installing manually:")
            print("pip install manim")

    @staticmethod
    def import_psutil() -> NoReturn:
        try:
            print("Installing psutil...")
            os.system(f"{sys.executable} -m pip install psutil")
        except Exception as e:
            print(f"Error installing/importing psutil: {e}")
