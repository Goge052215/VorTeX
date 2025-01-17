'''
Copyright (c) 2025 George Huang. All Rights Reserved.

This file is main part of the VorTeX Calculator project.

This file and its contents are protected under international copyright laws.
No part of this file may be reproduced, distributed, or transmitted in any form
or by any means, including photocopying, recording, or other electronic or
mechanical methods, without the prior written permission of the copyright owner.

PROPRIETARY AND CONFIDENTIAL
This file contains proprietary and confidential information. Unauthorized
copying, distribution, or use of this file, via any medium, is strictly
prohibited.

LICENSE RESTRICTIONS
- Commercial use is strictly prohibited without explicit written permission
- Modifications to this file are not permitted
- Distribution or sharing of this file is not permitted
- Private use must maintain all copyright and license notices

Version: 1.0.2
Last Updated: 2025.1
'''

import sys
import re
import os
import logging
from themes.theme_manager import ThemeManager, get_tokyo_night_theme, get_aura_theme, get_light_theme
from ui.legend_window import LegendWindow
from ui.ui_config import UiConfig
from ui.ui_components import UIComponents
from modules.modules_import import PackageImporter
from matlab_interface.sympy_to_matlab import SympyToMatlab
from latex_pack.latex_calculation import LatexCalculation
from matlab_interface.evaluate_expression import EvaluateExpression
import sympy as sy
from latex_pack.shortcut import ExpressionShortcuts
from matlab_interface.auto_simplify import AutoSimplify
from matlab_interface.display import Display
from ip_check.ip_check import IPCheck
from manim_visual.manim_visualizer import MathVisualizer
from ui.visualization_window import VisualizationWindow
from ui.settings_window import SettingsWindow
from sympy_pack.sympy_calculation import SympyCalculation

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    from PyQt5.QtWidgets import (
        QApplication, QWidget, QPushButton,
        QComboBox, QMessageBox, QTextEdit, QHBoxLayout, QGridLayout, 
        QScrollArea, QSpacerItem, QSizePolicy, QMenu, QInputDialog
    )
    from PyQt5.QtGui import QFont, QIcon
    from PyQt5.QtCore import Qt
except ImportError:
    PackageImporter.import_PyQt5()

try:
    import matlab.engine
except ImportError:
    PackageImporter.import_matlab()

try:
    import sympy as sy
    from sympy import sympify, Symbol
    import numpy as np
    from latex2sympy2 import latex2sympy
except ImportError:
    PackageImporter.import_sympy()

try:
    import manim
except ImportError:
    PackageImporter.import_manim()

from themes.theme_manager import (
    ThemeManager, get_tokyo_night_theme, 
    get_aura_theme, get_light_theme
)
from ui.legend_window import LegendWindow

FONT_NAME = "Monaspace Neon"
FONT_SIZE = 14

def _configure_logger():
    """Configure the root logger with minimal output."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    logging.getLogger('matlab_interface').setLevel(logging.WARNING)
    logging.getLogger('__main__').setLevel(logging.WARNING)

def _download_fonts():
    if IPCheck.ip_check():
        download_fonts()
    else:
        logger.warning("IP check failed, skipping font download")
        FONT_NAME = "Arial"

def download_fonts():
    try:
        if sys.platform == "darwin":  # macOS
            result = os.system("bash fonts/fonts_download.bash")
            if result != 0:
                logger.error("Failed to download fonts on macOS")
                return False
                
        elif sys.platform in ["win32", "win64"]:  # Windows
            result = os.system("powershell -ExecutionPolicy Bypass -File fonts/fonts_download_win.ps1")
            if result != 0:
                logger.error("Failed to download fonts on Windows")
                return False
                
        else:
            logger.error(f"Unsupported platform: {sys.platform}")
            print(f"Unsupported platform: {sys.platform}. Please install the fonts manually.")
            return False
            
        logger.info("Fonts downloaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading fonts: {str(e)}")
        print(f"Error downloading fonts: {str(e)}")
        return False

def parse_latex_expression(latex_expr):
    logger.debug(f"Original expression: '{latex_expr}'")

    latex_expr = ExpressionShortcuts.convert_combinatorial_expression(latex_expr)
    logger.debug(f"After combinatorial conversion: {latex_expr}")

    '''def add_multiplication(expr):
        # Don't modify content inside function calls and integrals
        parts = []
        last_end = 0
        stack = []
        in_integral = False
        
        i = 0
        while i < len(expr):
            # Check for integral expressions
            if expr[i:].startswith('int'):
                in_integral = True
                
            if expr[i] == '(':
                if not stack and not in_integral:
                    # Only add multiplication outside integrals
                    parts.append(re.sub(r'(\d+)\s+([a-zA-Z])', r'\1*\2', expr[last_end:i]))
                stack.append(i)
                
            elif expr[i] == ')':
                if stack:
                    if len(stack) == 1:
                        parts.append(expr[stack[0]:i+1])
                        last_end = i + 1
                        if 'int' in parts[-1]:
                            in_integral = False
                    stack.pop()
                    
            i += 1
        
        if last_end < len(expr):
            # Only add multiplication outside integrals
            if not in_integral:
                parts.append(re.sub(r'(\d+)\s+([a-zA-Z])', r'\1*\2', expr[last_end:]))
            else:
                parts.append(expr[last_end:])
        
        return ''.join(parts)

    latex_expr = add_multiplication(latex_expr)'''
    latex_expr = re.sub(r'(\d+)/(\d+)\s+([a-zA-Z])', r'(\1/\2)*\3', latex_expr)
    
    is_equation = '=' in latex_expr
    if is_equation:
        latex_expr = latex_expr.replace('==', '=').replace('=', '==')
        left_side, right_side = latex_expr.split('==')
        latex_expr = f"solve({left_side} - ({right_side}), x)"
        logger.debug(f"Converted equation to solve format: {latex_expr}")

    latex_expr = ExpressionShortcuts.convert_integral_expression(latex_expr)
    logger.debug(f"Converted integral expression: '{latex_expr}'")

    derivative_pattern = r'd/d([a-zA-Z])\s*([^$]+)'
    def replace_derivative(match):
        var = match.group(1)
        expr = match.group(2).strip()
        expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)
        expr = expr.replace('^', '.^')
        return f"diff({expr}, {var})"

    higher_derivative_pattern = r'd(\d+)/d([a-zA-Z])\^?(\d*)\s*([^$]+)'
    def replace_higher_derivative(match):
        order = match.group(1)
        var = match.group(2)
        expr = match.group(4).strip()
        expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)
        expr = expr.replace('^', '.^')
        return f"diff({expr}, {var}, {order})"

    latex_expr = re.sub(higher_derivative_pattern, replace_higher_derivative, latex_expr)
    latex_expr = re.sub(derivative_pattern, replace_derivative, latex_expr)

    latex_expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', latex_expr)
    logger.debug(f"After explicit multiplication: '{latex_expr}'")

    latex_expr = latex_expr.replace('ln(', 'log(')

    logger.debug(f"Final MATLAB expression: '{latex_expr}'")
    return latex_expr

class CalculatorApp(QWidget, LatexCalculation):
    """
    Main calculator application window with support for LaTeX, MATLAB, and Matrix operations.
    """
    def __init__(self):
        super().__init__()
        
        logger = logging.getLogger(__name__)
        logger.handlers.clear()
        
        self.current_theme = "aura"
        
        if not logger.handlers:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            file_handler = logging.FileHandler('calculator.log')
            file_handler.setFormatter(formatter)
            
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(stream_handler)
            logger.setLevel(logging.DEBUG)
        
        self.logger = logger
        self.logger.info("Initializing Calculator Application")
        
        self.theme_manager = ThemeManager()
        self.legend_window = None
        self.ui_components = UIComponents(self)
        self.ui_components.init_ui()
        self._init_theme()
        self.matrix_memory = {}

        self.result_display.setFont(QFont(FONT_NAME, FONT_SIZE))
        self.result_display.setWordWrap(True)
        self.result_display.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.display = Display(
            self.result_display,
            font_name=FONT_NAME,
            font_size=FONT_SIZE,
            bold=True
        ) 
        
        self.matlab_available = False
        try:
            self.eng = matlab.engine.start_matlab()
            self.evaluator = EvaluateExpression(self.eng)
            self.auto_simplify = AutoSimplify(self.eng)
            self.matlab_available = True
            self.logger.info("MATLAB engine started successfully")
        except Exception as e:
            self.logger.warning(f"MATLAB engine not available: {e}")
            self.logger.info("Switching to SymPy-only mode")
            QMessageBox.warning(
                self,
                "MATLAB Unavailable",
                "MATLAB engine could not be started. The calculator will run in SymPy-only mode.\n"
                "Some features may be limited."
            )
            self.sympy_calculator = SympyCalculation()

        self.sympy_converter = SympyToMatlab()
        self.current_log_type = None
        self.visualizer = MathVisualizer()
        
        self.viz_button = QPushButton("Visualize")
        self.viz_button.clicked.connect(self.handle_visualization)
        
    def _init_theme(self):
        self.set_theme("aura")

    def show_settings(self):
        if not hasattr(self, 'settings_window'):
            self.settings_window = SettingsWindow(self)
        self.settings_window.show()

    def set_theme(self, theme):
        theme_getters = {
            "tokyo_night": get_tokyo_night_theme,
            "aura": get_aura_theme,
            "light": get_light_theme
        }
        
        theme_getter = theme_getters.get(theme)
        if not theme_getter:
            return
        
        theme_data = theme_getter()
        
        self.setStyleSheet(theme_data["main_widget"])
        self.theme_button.setStyleSheet(theme_data["theme_button"])
        self.legend_button.setStyleSheet(theme_data["theme_button"])
        
        if self.legend_window:
            self.legend_window.update_colors(
                theme_data["text_color"], 
                theme_data["border_color"]
            )

    def show_legend(self):
        if self.legend_window is None:
            self.legend_window = LegendWindow()
        self.legend_window.show()

    def on_mode_changed(self, mode):
        if not self.matlab_available and mode in ['LaTeX', 'MATLAB', 'Matrix']:
            QMessageBox.information(
                self,
                "Mode Unavailable",
                f"{mode} mode requires MATLAB. Please use SymPy mode or install MATLAB."
            )
            self.combo_mode.setCurrentText('SymPy')
            return

        try:
            ui_config = UiConfig()
            ui_config.mode_config(self)
            
            config = ui_config.mode_configs.get(mode, ui_config.mode_configs['SymPy'])
            
            for widget in config['show']:
                widget.show()
            for widget in config['hide']:
                widget.hide()
            
            height, width = config['dimensions']
            self.setFixedHeight(height)
            self.setFixedWidth(width)

            placeholders = {
                'LaTeX': 'Enter Simplified LaTeX expression, e.g., 5C2 + sin(pi/2)',
                'MATLAB': 'Enter MATLAB expression, e.g., nchoosek(5,2) + sin(pi/2)',
                'SymPy': 'Enter Simplified LaTeX expression, e.g., 5C2 + sin(pi/2)',
                'Matrix': 'Enter matrix in format: [1 2; 3 4] ([row 1; row 2; ...])'
            }
            
            self.entry_formula.setPlaceholderText(placeholders.get(mode, placeholders['SymPy']))
            
        except Exception as e:
            self.logger.error(f"Error in mode change: {e}")
            QMessageBox.critical(self, "Error", f"Error changing mode: {str(e)}")

    def store_matrix(self):
        matrix_text = self.matrix_input.toPlainText().strip()
        if not matrix_text:
            QMessageBox.warning(self, "Input Error", "Please enter a matrix.")
            return
        
        if not (matrix_text.startswith('[') and matrix_text.endswith(']')):
            QMessageBox.warning(self, "Format Error", "Matrix must be enclosed in square brackets []")
            return
        
        if ';' not in matrix_text and matrix_text.count('[') == 1:
            QMessageBox.warning(self, "Format Error", "For matrices with multiple rows, use semicolons to separate rows.\nExample: [1 2; 3 4]")
            return

        name, ok = QInputDialog.getText(self, 'Store Matrix', 'Enter a name for this matrix:')
        if ok and name:
            try:
                validation_cmd = (
                    f"try, "
                    f"temp_matrix = {matrix_text}; "
                    f"[rows, cols] = size(temp_matrix); "
                    f"if ~isnumeric(temp_matrix), error('Matrix must contain only numbers'); end; "
                    f"catch ME, "
                    f"disp(ME.message); "
                    f"error(ME.message); "
                    f"end"
                )
                
                self.eng.eval(validation_cmd, nargout=0)
                self.matrix_memory[name] = matrix_text
                QMessageBox.information(self, "Success", f"Matrix '{name}' stored successfully.")
                
            except matlab.engine.MatlabExecutionError as me:
                error_msg = str(me).split('\n')[-1]
                QMessageBox.critical(self, "MATLAB Error", f"Invalid matrix: {error_msg}\n\nPlease ensure:\n- Matrix is properly formatted\n- All elements are numbers\n- Rows have equal length")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
    
    def recall_matrix(self):
        if not self.matrix_memory:
            QMessageBox.warning(self, "No Stored Matrices", "There are no matrices stored in memory.")
            return
        
        name, ok = QInputDialog.getItem(self, 'Recall Matrix', 'Select a matrix to recall:', 
                                        list(self.matrix_memory.keys()), 0, False)
        if ok and name:
            self.matrix_input.setPlainText(self.matrix_memory[name])
    
    def sympy_to_matlab(self, expr):
        try:
            return self.sympy_converter.sympy_to_matlab(expr)
        except Exception as e:
            self.logger.error(f"Error in sympy_to_matlab conversion: {e}", exc_info=True)
            raise

    def _handle_list_expression(self, expr):
        self.logger.debug(f"Handling list expression: {expr}")
        try:
            if isinstance(expr, list):
                if not expr:
                    self.logger.warning("Empty expression list encountered")
                    raise ValueError("Empty expression list")
                return expr[0]
            return expr
        except Exception as e:
            self.logger.error(f"Error handling list expression: {e}")
            raise

    def _handle_integral(self, expr):
        self.logger.debug(f"Handling integral expression: {expr}")
        try:
            func = self.sympy_to_matlab(expr.function)
            var = expr.variables[0]
            
            self.logger.debug(f"Integral function: {func}, variable: {var}")
            
            func = self._convert_logarithms(func)
            
            if len(expr.limits) == 0:
                result = f"int({func}, {var})"
                self.logger.debug(f"Created indefinite integral: {result}")
                return result
            
            if len(expr.limits) == 1:
                if len(expr.limits[0]) == 3:
                    a, b = expr.limits[0][1], expr.limits[0][2]
                    result = f"int({func}, {var}, {a}, {b})"
                    self.logger.debug(f"Created definite integral: {result}")
                    return result
                result = f"int({func}, {var})"
                self.logger.debug(f"Created integral with variable: {result}")
                return result
            
            self.logger.error("Unsupported integral type encountered")
            raise ValueError("Unsupported integral type")
            
        except Exception as e:
            self.logger.error(f"Error handling integral: {e}")
            raise

    def _handle_derivative(self, expr):
        self.logger.debug(f"Handling derivative expression: {expr}")
        try:
            func = self.sympy_to_matlab(expr.expr)
            
            var = str(expr.variables[0])
            
            if hasattr(expr, 'order'):
                order = expr.order
            else:
                order = sum(1 for v in expr.variables if str(v) == var)
            
            func = self._convert_logarithms(func)
            
            if order > 1:
                result = f"diff({func}, {var}, {order})"
            else:
                result = f"diff({func}, {var})"
            
            self.logger.debug(f"Created derivative expression: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error handling derivative: {e}", exc_info=True)
            raise

    def _process_derivative_variables(self, variables):
        self.logger.debug(f"Processing derivative variables: {variables}")
        try:
            matlab_vars = []
            last_var = None
            order = 0
            
            for var in variables:
                if var == last_var:
                    order += 1
                else:
                    if last_var is not None:
                        matlab_vars.append(f"{str(last_var)}, {order}" if order > 1 else str(last_var))
                    last_var = var
                    order = 1
                    
            if last_var is not None:
                matlab_vars.append(f"{str(last_var)}, {order}" if order > 1 else str(last_var))
                
            self.logger.debug(f"Processed variables: {matlab_vars}")
            return matlab_vars
            
        except Exception as e:
            self.logger.error(f"Error processing derivative variables: {e}")
            raise

    def _handle_equation(self, expr):
        self.logger.debug(f"Handling equation expression: {expr}")
        try:
            lhs = self.sympy_to_matlab(expr.lhs)
            rhs = self.sympy_to_matlab(expr.rhs)
            result = f"{lhs} == {rhs}"
            self.logger.debug(f"Created equation: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Error handling equation: {e}")
            raise

    def _handle_function(self, expr):
        self.logger.debug(f"Handling function expression: {expr}")
        try:
            func_name = expr.func.__name__
            args = [self.sympy_to_matlab(arg) for arg in expr.args]
            
            function_handlers = {
                'sin': lambda args: f"sind({args[0]})" if self.combo_angle.currentText() == 'Degree' else f"sin({args[0]})",
                'cos': lambda args: f"cosd({args[0]})" if self.combo_angle.currentText() == 'Degree' else f"cos({args[0]})",
                'tan': lambda args: f"tand({args[0]})" if self.combo_angle.currentText() == 'Degree' else f"tan({args[0]})",
                'csc': lambda args: f"csc({args[0]})",
                'sec': lambda args: f"sec({args[0]})",
                'cot': lambda args: f"cot({args[0]})",
                
                'asin': lambda args: f"asind({args[0]})" if self.combo_angle.currentText() == 'Degree' else f"asin({args[0]})",
                'arcsin': lambda args: f"asind({args[0]})" if self.combo_angle.currentText() == 'Degree' else f"asin({args[0]})",
                'acos': lambda args: f"acosd({args[0]})" if self.combo_angle.currentText() == 'Degree' else f"acos({args[0]})",
                'arccos': lambda args: f"acosd({args[0]})" if self.combo_angle.currentText() == 'Degree' else f"acos({args[0]})",
                'atan': lambda args: f"atand({args[0]})" if self.combo_angle.currentText() == 'Degree' else f"atan({args[0]})",
                'arctan': lambda args: f"atand({args[0]})" if self.combo_angle.currentText() == 'Degree' else f"atan({args[0]})",
                
                'sinh': lambda args: f"sinh({args[0]})",
                'cosh': lambda args: f"cosh({args[0]})",
                'tanh': lambda args: f"tanh({args[0]})",
                'csch': lambda args: f"csch({args[0]})",
                'sech': lambda args: f"sech({args[0]})",
                'coth': lambda args: f"coth({args[0]})",
                
                'asinh': lambda args: f"asinh({args[0]})",
                'arcsinh': lambda args: f"asinh({args[0]})",
                'acosh': lambda args: f"acosh({args[0]})",
                'arccosh': lambda args: f"acosh({args[0]})",
                'atanh': lambda args: f"atanh({args[0]})",
                'arctanh': lambda args: f"atanh({args[0]})",
                
                'log': lambda args: f"log({args[0]})",
                'sqrt': lambda args: f"sqrt({args[0]})",
                'Abs': lambda args: f"abs({args[0]})",
                'factorial': lambda args: f"factorial({args[0]})",
                'nchoosek': lambda args: f"nchoosek({args[0]}, {args[1]})",
                'symsum': lambda args: f"symsum({args[0]}, {args[1]}, {args[2]}, {args[3]})",
                'prod': lambda args: f"prod({args[0]}, {args[1]}, {args[2]}, {args[3]})"
            }
            
            if func_name in function_handlers:
                result = function_handlers[func_name](args)
                self.logger.debug(f"Handled special function {func_name}: {result}")
                return result
            
            result = f"{func_name}({', '.join(args)})"
            self.logger.debug(f"Handled generic function {func_name}: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error handling function: {e}")
            raise

    def _process_expression_string(self, expr):
        expr_str = str(expr)
        
        expr_str = self._convert_logarithms(expr_str)
        expr_str = self._apply_function_replacements(expr_str)
        expr_str = self._handle_integral_derivative_patterns(expr_str)
        expr_str = self._process_operators(expr_str)
        expr_str = self._handle_combinations_permutations(expr_str)
        
        return expr_str

    def _convert_logarithms(self, expr_str, for_display=False):
        """
        Convert different types of logarithms between display and MATLAB formats.
        
        Args:
            expr_str (str): The mathematical expression string.
            for_display (bool): If True, convert 'log' to 'ln' for user display.
                                 If False, convert 'ln' to 'log' for MATLAB evaluation.
                                 
        Returns:
            str: The converted expression string.
        """
        if for_display:
            expr_str = re.sub(r'\blog\s*\(', 'ln(', expr_str)
        else:
            expr_str = re.sub(r'\bln\s*\(', 'log(', expr_str)
            expr_str = re.sub(r'\blog\s*\(\s*x\s*,\s*10\s*\)', 'log10(x)', expr_str)
        
        return expr_str

    def _apply_function_replacements(self, expr_str):
        replacements = {
            'ln': 'log', 'log10': 'log10', 'log2': 'log2',
            'sin': 'sin', 'cos': 'cos', 'tan': 'tan',
            'asin': 'asin', 'acos': 'acos', 'atan': 'atan',
            'log': 'log', 'exp': 'exp', 'sqrt': 'sqrt',
            'factorial': 'factorial', '**': '^', 'pi': 'pi',
            'E': 'exp(1)', 'Abs': 'abs', 'Derivative': 'diff'
        }
        
        for sympy_func, matlab_func in replacements.items():
            expr_str = expr_str.replace(sympy_func, matlab_func)
        return expr_str

    def _handle_integral_derivative_patterns(self, expr_str):
        patterns = {
            r'Integral\((.*?), \((.*?), (.*?), (.*?)\)\)': r'integral(@(x) \1, \3, \4)',
            r'Integral\((.*?), (.*?)\)': r'int(\1, \2)',
            r'Function\((.*?)\)': r'\1'
        }
        
        for pattern, replacement in patterns.items():
            expr_str = re.sub(pattern, replacement, expr_str)
        return expr_str

    def _process_operators(self, expr_str):
        expr_str = expr_str.replace('**', '.^')
        expr_str = expr_str.replace('/', './').replace('*', '.*')
        expr_str = expr_str.replace('.*1', '').replace('.*0', '0')
        return expr_str

    def _handle_combinations_permutations(self, expr_str):
        patterns = {
            r'\\binom\s*\{\s*(\d+)\s*\}\s*\{\s*(\d+)\s*\}': r'nchoosek(\1, \2)',
            r'\\choose\s*\{\s*(\d+)\s*\}\s*\{\s*(\d+)\s*\}': r'nchoosek(\1, \2)',
            r'\bnCr\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)': r'nchoosek(\1, \2)',
            r'\bnPr\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)': r'nchoosek(\1, \2)*factorial(\2)'
        }
        
        for pattern, replacement in patterns.items():
            expr_str = re.sub(pattern, replacement, expr_str, flags=re.IGNORECASE)
        return expr_str

    def calculate(self):
        try:
            mode = self.combo_mode.currentText()
            angle_mode = self.combo_angle.currentText()
            
            if not self.matlab_available and mode in ['LaTeX', 'MATLAB', 'Matrix']:
                QMessageBox.information(
                    self,
                    "Mode Unavailable",
                    f"{mode} mode requires MATLAB. Please use SymPy mode or install MATLAB."
                )
                self.combo_mode.setCurrentText('SymPy')
                return
            
            if mode == 'LaTeX':
                self.handle_latex_calculation(angle_mode)
            elif mode == 'MATLAB':
                self.handle_matlab_calculation(angle_mode)
            elif mode == 'SymPy':
                self.handle_sympy_calculation(angle_mode)
            elif mode == 'Matrix':
                self.calculate_matrix()
                
        except Exception as e:
            self.logger.error(f"Error in calculate: {e}")
            QMessageBox.critical(self, "Error", f"Error during calculation: {str(e)}")

    def handle_sympy_calculation(self, angle_mode):
        expression = self.entry_formula.toPlainText().strip()
        if not expression:
            QMessageBox.warning(self, "Input Error", "Please enter an expression.")
            return

        try:
            calculator = SympyCalculation()
            result = calculator.evaluate(expression, angle_mode)
            
            self.result_display.setText(result)
            self.logger.debug(f"SymPy calculation result: {result}")

        except Exception as e:
            self.logger.error(f"Error in SymPy calculation: {e}")
            QMessageBox.critical(self, "Error", f"Error evaluating expression: {str(e)}")

    def format_matrix_result(self, result, operation):
        if result is None:
            return "Operation Failed."

        if isinstance(result, list):
            return str(result)
        else:
            return str(result)
        
    # Deprecated, will be removed soon
    '''
    def handle_matrix_calculation(self):
        matrix_text = self.matrix_input.toPlainText().strip()
        operation = self.combo_matrix_op.currentText()
        
        if not matrix_text:
            QMessageBox.critical(self, "Input Error", "Please enter a matrix.")
            return
        
        if matrix_text in self.matrix_memory:
            matrix_text = self.matrix_memory[matrix_text]

        try:
            self.eng.eval(f"matrix = {matrix_text};", nargout=0)
        except matlab.engine.MatlabExecutionError as me:
            QMessageBox.critical(self, "MATLAB Error", f"Error evaluating matrix: {me}")
            return
        
        # Give appropriate roundings for specific mode
        try:
            if operation == 'Determinant':
                result = self.eng.eval("det(matrix)", nargout=1)
                result = round(float(result), 2)

            elif operation == 'Inverse':
                result = self.eng.eval("inv(matrix)", nargout=1)
                result = np.round(np.array(result), 2).tolist()

            elif operation == 'Eigenvalues':
                result = self.eng.eval("eig(matrix)", nargout=1)
                result = np.round(np.array(result), 2).tolist()

            elif operation == 'Rank':
                result = self.eng.eval("rank(matrix)", nargout=1)
                result = int(result)

            elif operation in ['Multiply', 'Add', 'Subtract', 'Divide']:
                result = self.handle_matrix_arithmetic(operation)

            elif operation == 'Differentiate':
                result = self.handle_matrix_differentiation()

            else:
                QMessageBox.warning(self, "Operation Warning", f"Operation '{operation}' is not supported.")
                return

            result = self.format_matrix_result(result, operation)

            self.result_label.setText(f"Result: {result}")
            self.result_label.setFont(QFont("Arial", 13, QFont.Bold))

        except matlab.engine.MatlabExecutionError as me:
            QMessageBox.critical(self, "MATLAB Error", f"MATLAB Error: {me}")
    '''

    def handle_matrix_arithmetic(self, operation):
        operation_map = {
            'Multiply': '*',
            'Add': '+',
            'Subtract': '-',
            'Divide': '/'
        }

        operation_symbol = operation_map.get(operation)
        second_matrix, ok = QInputDialog.getText(
            self, 
            f'Matrix {operation}', 
            f'Enter the second matrix or its name for {operation}:'
        )
        if ok and second_matrix:
            if second_matrix in self.matrix_memory:
                second_matrix = self.matrix_memory[second_matrix]
            try:
                self.eng.eval(f"matrix2 = {second_matrix};", nargout=0)
                expr = f"matrix {operation_symbol} matrix2"
                result = self.eng.eval(f"result = {expr};", nargout=1)
                result = self.eng.eval("result", nargout=1)
                result = np.round(np.array(result), 2).tolist()
                return result
            except matlab.engine.MatlabExecutionError as me:
                QMessageBox.critical(self, "MATLAB Error", f"Error performing '{operation}': {me}")
                return None
        else:
            QMessageBox.information(self, "Operation Cancelled", "Matrix operation was cancelled.")
            return None

    def handle_matrix_differentiation(self):
        variable, ok = QInputDialog.getText(
            self, 
            'Differentiate Matrix', 
            'Enter the variable to differentiate with respect to:'
        )
        if ok and variable:
            try:
                result = self.eng.eval(f"diff(matrix, {variable})", nargout=1)
                result = np.round(np.array(result), 2).tolist()
                return result
            except matlab.engine.MatlabExecutionError as me:
                QMessageBox.critical(self, "MATLAB Error", f"Error in differentiation: {me}")
                return None
        else:
            QMessageBox.information(self, "Operation Cancelled", "Differentiation was cancelled.")
            return None
    
    def handle_latex_calculation(self, angle_mode):
        """Handle LaTeX input and ensure correct MATLAB processing."""
        expression = self.entry_formula.toPlainText().strip()
        if not expression:
            QMessageBox.warning(self, "Input Error", "Please enter an expression.")
            return

        try:
            self.logger.debug(f"Original expression: '{expression}'")
                
            sympy_expr = parse_latex_expression(expression)
            self.logger.debug(f"Converted to SymPy expression: '{sympy_expr}'")

            sympy_to_matlab_converter = SympyToMatlab()
            matlab_expression = sympy_to_matlab_converter.sympy_to_matlab(sympy_expr)
            self.logger.debug(f"Converted to MATLAB expression: '{matlab_expression}'")

            matlab_expression = ExpressionShortcuts.convert_shortcut(matlab_expression)
            self.logger.debug(f"After shortcut conversion: {matlab_expression}")

            if angle_mode == 'Degree' and 'limit' not in matlab_expression:
                matlab_expression = re.sub(r'\bsin\((.*?)\)', lambda m: f"sind({m.group(1)})", matlab_expression)
                matlab_expression = re.sub(r'\bcos\((.*?)\)', lambda m: f"cosd({m.group(1)})", matlab_expression)
                matlab_expression = re.sub(r'\btan\((.*?)\)', lambda m: f"tand({m.group(1)})", matlab_expression)

            self.logger.debug(f"Final MATLAB expression: '{matlab_expression}'")

            result = self.evaluator.evaluate_matlab_expression(matlab_expression)
            self.logger.debug(f"Raw result from MATLAB: '{result}'")

            # Check if the result is a numeric string
            try:
                float(result)
                is_numeric = True
            except ValueError:
                is_numeric = False

            # Format the result based on its type
            if is_numeric:
                displayed_result = f"{float(result):.3f}"
            else:
                displayed_result = result

            displayed_result = displayed_result.replace('log(', 'ln(')
            displayed_result = displayed_result.replace('cosd(', 'cos(')
            displayed_result = displayed_result.replace('sind(', 'sin(')
            displayed_result = displayed_result.replace('tand(', 'tan(')

            if displayed_result.count('(') > displayed_result.count(')'):
                displayed_result += ')'
            
            displayed_result = self.evaluator._postprocess_result(displayed_result)

            self.result_display.setText(displayed_result)
            self.logger.debug(f"Displayed Result: {displayed_result}")

        except Exception as e:
            self.logger.error(f"Error in latex calculation: {e}")
            QMessageBox.critical(self, "Error", f"Error evaluating expression: {str(e)}")

    def evaluate_expression(self, matlab_expression):
        return self.evaluator.evaluate_matlab_expression(matlab_expression)

    def _is_numeric_expression(self, sympy_expr):
        return sympy_expr.is_number

    def _extract_function_argument(self, expression):
        match = re.match(r'([a-zA-Z]+)\s*\(\s*([^\)]+)\s*\)', expression)
        if match:
            func, arg = match.groups()
            return func, arg
        return '', ''

    def _parse_expression(self, expression):
        """
        Parse the input expression into a SymPy expression.
        
        Args:
            expression (str): The input expression string.
        
        Returns:
            sympy.Expr: The parsed SymPy expression.
        """
        try:
            processed_expr = EvaluateExpression.preprocess_expression(expression)
            self.logger.debug(f"Processed expression for SymPy: {processed_expr}")
            
            sympy_expr = sy.sympify(processed_expr, evaluate=False)
            return sympy_expr

        except Exception as e:
            self.logger.error(f"SymPy parsing error: {e}")
            raise ValueError(f"Invalid expression format: {e}") from e

    def handle_matlab_calculation(self):
        input_text = self.entry_formula.toPlainText().strip()
        
        if not input_text:
            QMessageBox.warning(self, "Input Error", "Please enter a MATLAB expression.")
            return

        try:
            input_text = ExpressionShortcuts.convert_shortcut(input_text)
            
            if self.combo_angle.currentText() == 'Degree':
                input_text = re.sub(r'\bsin\((.*?)\)', lambda m: f"sind({m.group(1)})", input_text)
                input_text = re.sub(r'\bcos\((.*?)\)', lambda m: f"cosd({m.group(1)})", input_text)
                input_text = re.sub(r'\btan\((.*?)\)', lambda m: f"tand({m.group(1)})", input_text)

            input_text = input_text.replace('ln(', 'log(')

            self.eng.eval(f"result = {input_text};", nargout=0)

            result = self.eng.eval("result", nargout=1)

            if isinstance(result, matlab.object):
                result = self.eng.eval("char(result)", nargout=1)
            elif isinstance(result, (int, float)):
                result = f"{result:.4f}"
            else:
                result = str(result)

            result = self.evaluator._postprocess_result(result)

            if 'inf' in result:
                result = result.replace('inf', '∞')

            result = result.replace('log(', 'ln(')
            result = result.replace('==', '=')

            self.result_label.setText(f"Result: {result}")
            self.result_label.setFont(QFont(FONT_NAME, FONT_SIZE))

        except matlab.engine.MatlabExecutionError as me:
            QMessageBox.critical(self, "MATLAB Error", f"MATLAB Error: {me}")
        except Exception as e:
            QMessageBox.critical(self, "Unexpected Error", f"Unexpected Error: {str(e)}")
    
    def closeEvent(self, event):
        try:
            if self.matlab_available:
                self.eng.quit()
                self.logger.info("MATLAB engine terminated")
        except Exception as e:
            self.logger.warning(f"Error terminating MATLAB engine: {e}")
        event.accept()

    def calculate_matrix(self):
        try:
            matrix_text = self.matrix_input.toPlainText().strip()
            operation = self.combo_matrix_op.currentText()
            
            if not matrix_text:
                QMessageBox.warning(self, "Input Error", "Please enter a matrix.")
                return
            
            if matrix_text in self.matrix_memory:
                matrix_text = self.matrix_memory[matrix_text]

            self.eng.eval(f"matrix = {matrix_text};", nargout=0)
            
            if operation == 'Determinant':
                result = self.eng.eval("det(matrix)", nargout=1)
                result = round(float(result), 2)

            elif operation == 'Inverse':
                result = self.eng.eval("inv(matrix)", nargout=1)
                result = np.round(np.array(result), 2).tolist()

            elif operation == 'Eigenvalues':
                result = self.eng.eval("eig(matrix)", nargout=1)
                result = np.round(np.array(result), 2).tolist()

            elif operation == 'Rank':
                result = self.eng.eval("rank(matrix)", nargout=1)
                result = int(result)

            elif operation in ['Multiply', 'Add', 'Subtract', 'Divide']:
                result = self.handle_matrix_arithmetic(operation)

            elif operation == 'Differentiate':
                result = self.handle_matrix_differentiation()

            else:
                QMessageBox.warning(self, "Operation Warning", f"Operation '{operation}' is not supported.")
                return

            result_str = self.format_matrix_result(result, operation)

            self.result_display.setText(result_str)
            self.result_display.setFont(QFont(FONT_NAME, FONT_SIZE))

        except matlab.engine.MatlabExecutionError as me:
            QMessageBox.critical(self, "MATLAB Error", f"MATLAB Error: {me}")
        except Exception as e:
            QMessageBox.critical(self, "Unexpected Error", f"Unexpected Error: {str(e)}")

    def handle_visualization(self):
        """Handle visualization of the current expression."""
        try:
            expr = self.entry_formula.toPlainText().strip()
            if not expr:
                QMessageBox.warning(self, "Error", "Please enter an expression to visualize.")
                return
                
            expr = expr.replace('ln', 'log') 
            expr = re.sub(r'(\d+)\s*([a-zA-Z])', r'\1*\2', expr) 
            expr = re.sub(r'\s+', '', expr) 
            
            if "==" in expr:
                expr = expr.split("==")[0]
            
            if "d/dx" in expr or "diff" in expr:
                QMessageBox.information(self, "Info", "Derivative visualization not yet implemented")
                return
            
            self.logger.debug(f"Visualizing expression: {expr}")
            
            # Create or show visualization window
            if not hasattr(self, 'viz_window') or self.viz_window is None:
                self.viz_window = VisualizationWindow(self)
                self.viz_window.show()
                self.viz_window.raise_()
                self.viz_window.activateWindow()
                self.viz_window.visualize_function(expr)
                
        except Exception as e:
            QMessageBox.critical(self, "Visualization Error", str(e))
            
    def convert_to_python_expr(self, expr: str) -> str:
        try:
            if expr.startswith('(-') and ')^' in expr:
                base, power = expr.split(')^')
                base = base[1:]
                if '/' in base:
                    num, den = map(float, base.split('/'))
                    decimal_base = num / den
                    return f"({decimal_base})**{power}"
            
            replacements = {
                'sind': 'np.sin',
                'cosd': 'np.cos',
                'tand': 'np.tan',
                'sin': 'np.sin',
                'cos': 'np.cos',
                'tan': 'np.tan',
                'exp': 'np.exp',
                'log': 'np.log',
                'sqrt': 'np.sqrt',
                '^': '**',
                'pi': 'np.pi',
                'e': 'np.e'
            }
            
            for old, new in replacements.items():
                expr = re.sub(rf'\b{old}\b', new, expr)
            
            return expr
            
        except Exception as e:
            self.logger.error(f"Error converting expression: {e}")
            return expr

if __name__ == '__main__':
    _configure_logger()
    _download_fonts()
    app = QApplication(sys.argv)
    calculator = CalculatorApp()
    calculator.show()
    sys.exit(app.exec_())
