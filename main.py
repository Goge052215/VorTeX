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

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    from PyQt5.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QLabel, QPushButton,
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

from themes.theme_manager import (
    ThemeManager, get_tokyo_night_theme, 
    get_aura_theme, get_light_theme
)
from ui.legend_window import LegendWindow

def download_fonts():
    """
    Download and install fonts based on the operating system.
    Returns True if successful, False otherwise.
    """
    '''try:
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
        return False'''

def parse_latex_expression(latex_expr):
    """
    Parse a LaTeX-like expression and convert it to a MATLAB-compatible expression.

    Args:
        latex_expr (str): The LaTeX-like expression.

    Returns:
        str: The MATLAB-compatible expression.
    """
    logger.debug(f"Original expression: '{latex_expr}'")

    # Handle limit expressions
    limit_pattern = r'lim\s*\(([a-zA-Z])\s*to\s*([^)]+)\)\s*(.+)'
    def replace_limit(match):
        var = match.group(1)
        limit_value = match.group(2)
        expr = match.group(3)
        return f"limit({expr}, {var}, {limit_value})"
    
    latex_expr = re.sub(limit_pattern, replace_limit, latex_expr)
    
    # Convert \infty and infty to inf for MATLAB processing
    latex_expr = latex_expr.replace(r'\infty', 'inf').replace('infty', 'inf')
    latex_expr = latex_expr.replace(r'-\infty', '-inf').replace('-infty', '-inf')
    
    # Check if it's an equation (contains '=')
    is_equation = '=' in latex_expr
    if is_equation:
        # Convert single '=' to '==' for MATLAB equation solving
        # But first ensure we don't double up existing '=='
        latex_expr = latex_expr.replace('==', '=').replace('=', '==')
        # Move everything to left side of equation
        left_side, right_side = latex_expr.split('==')
        latex_expr = f"solve({left_side} - ({right_side}), x)"
        logger.debug(f"Converted equation to solve format: {latex_expr}")
    
    # Handle combinatorial expressions (binomial coefficients)
    binom_pattern = r'binom\{(\d+)\}\{(\d+)\}'
    def replace_binom(match):
        n = match.group(1)
        k = match.group(2)
        return f"nchoosek({n}, {k})"
    
    latex_expr = re.sub(binom_pattern, replace_binom, latex_expr)
    logger.debug(f"After binom conversion: '{latex_expr}'")

    # Preprocess integral expressions
    latex_expr = ExpressionShortcuts.convert_integral_expression(latex_expr)
    logger.debug(f"Converted integral expression: '{latex_expr}'")

    # Handle derivative expressions directly
    derivative_pattern = r'd/d([a-zA-Z])\s*([^$]+)'
    def replace_derivative(match):
        var = match.group(1)
        expr = match.group(2).strip()
        expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)
        expr = expr.replace('^', '.^')
        return f"diff({expr}, {var})"

    # Handle higher-order derivatives
    higher_derivative_pattern = r'd(\d+)/d([a-zA-Z])\^?(\d*)\s*([^$]+)'
    def replace_higher_derivative(match):
        order = match.group(1)
        var = match.group(2)
        expr = match.group(4).strip()
        expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)
        expr = expr.replace('^', '.^')
        return f"diff({expr}, {var}, {order})"

    # Try higher-order derivative first, then first-order derivative
    latex_expr = re.sub(higher_derivative_pattern, replace_higher_derivative, latex_expr)
    latex_expr = re.sub(derivative_pattern, replace_derivative, latex_expr)

    # Ensure multiplication is explicit for remaining terms
    latex_expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', latex_expr)
    logger.debug(f"After explicit multiplication: '{latex_expr}'")

    # Correctly convert 'ln' to 'log' for MATLAB compatibility
    latex_expr = latex_expr.replace('ln(', 'log(')

    logger.debug(f"Final MATLAB expression: '{latex_expr}'")
    return latex_expr

class CalculatorApp(QWidget, LatexCalculation):
    """
    Main calculator application window with support for LaTeX, MATLAB, and Matrix operations.
    """
    def __init__(self):
        """Initialize the calculator application."""
        super().__init__()
        
        # Configure logging - modified to prevent duplicate logs
        logger = logging.getLogger(__name__)
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Only add handlers if they don't exist
        if not logger.handlers:
            # Configure logging format
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            # File handler
            file_handler = logging.FileHandler('calculator.log')
            file_handler.setFormatter(formatter)
            
            # Stream handler
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            
            # Add handlers and set level
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

        self.result_display.setFont(QFont("Monaspace Neon", 14))
        self.result_display.setWordWrap(True)
        self.result_display.setTextInteractionFlags(Qt.TextSelectableByMouse)  # Make text selectable

        self.display = Display(
            self.result_display,
            font_name="Monaspace Neon",
            font_size=14,
            bold=True
        ) 
        
        try:
            self.eng = matlab.engine.start_matlab()
            self.eng.eval("syms x f(x)", nargout=0)
            self.simplifier = AutoSimplify(self.eng)
            self.logger.info("MATLAB engine started successfully")
        except Exception as e:
            self.logger.error(f"Failed to start MATLAB engine: {e}")

        self.sympy_converter = SympyToMatlab()

        self.current_log_type = None

        self.evaluator = EvaluateExpression(self.eng)

    def _init_theme(self):
        """Initialize and set default theme."""
        self.set_theme("aura")


    def show_theme_menu(self):
        menu = QMenu(self)
        aura_action = menu.addAction("Aura")
        dark_action = menu.addAction("Tokyo Night")
        light_action = menu.addAction("Light")
        
        aura_action.triggered.connect(lambda: self.set_theme("aura"))
        dark_action.triggered.connect(lambda: self.set_theme("tokyo_night"))
        light_action.triggered.connect(lambda: self.set_theme("light"))
        
        menu.exec_(self.theme_button.mapToGlobal(self.theme_button.rect().bottomLeft()))
    
    def set_theme(self, theme):
        # Theme getter mapping
        theme_getters = {
            "tokyo_night": get_tokyo_night_theme,
            "aura": get_aura_theme,
            "light": get_light_theme
        }
        
        # Get theme data using the mapping
        theme_getter = theme_getters.get(theme)
        if not theme_getter:
            return  # Invalid theme name
        
        theme_data = theme_getter()
        
        # Apply theme styles in one go
        self.setStyleSheet(theme_data["main_widget"])
        self.theme_button.setStyleSheet(theme_data["theme_button"])
        self.legend_button.setStyleSheet(theme_data["theme_button"])
        
        # Update legend window if it exists
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
        """Handle UI changes when calculator mode is changed.
        
        Args:
            mode (str): The selected mode ('Matrix' or 'LaTeX/MATLAB')
        """
        try:
            # Create UiConfig instance and initialize mode configurations
            ui_config = UiConfig()
            ui_config.mode_config(self)  # Pass the calculator instance
            
            # Get configuration for current mode (use default if not Matrix)
            config = ui_config.mode_configs.get(mode, ui_config.mode_configs['default'])
            
            # Apply visibility changes
            for widget in config['show']:
                widget.show()
            for widget in config['hide']:
                widget.hide()
            
            # Always show the Calculate button
            self.calculate_matrix_button.show()
            
            # Set window dimensions
            height, width = config['dimensions']
            
            self.setFixedHeight(height)
            self.setFixedWidth(width)
            
        except AttributeError as e:
            print(f"Error in on_mode_changed: {e}")
            print(f"self.matrix_input: {self.matrix_input}")
            print(f"self.entry_formula: {self.entry_formula}")
    
    def store_matrix(self):
        matrix_text = self.matrix_input.toPlainText().strip()
        if not matrix_text:
            QMessageBox.warning(self, "Input Error", "Please enter a matrix.")
            return
        
        # Check if the matrix text is properly formatted
        if not (matrix_text.startswith('[') and matrix_text.endswith(']')):
            QMessageBox.warning(self, "Format Error", "Matrix must be enclosed in square brackets []")
            return
        
        # Check for proper matrix format with semicolons
        if ';' not in matrix_text and matrix_text.count('[') == 1:
            QMessageBox.warning(self, "Format Error", "For matrices with multiple rows, use semicolons to separate rows.\nExample: [1 2; 3 4]")
            return

        name, ok = QInputDialog.getText(self, 'Store Matrix', 'Enter a name for this matrix:')
        if ok and name:
            try:
                # Try to create and validate the matrix
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
                error_msg = str(me).split('\n')[-1]  # Get the last line of the error message
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
        """Convert SymPy expression to MATLAB format using the SympyToMatlab class."""
        try:
            return self.sympy_converter.sympy_to_matlab(expr)
        except Exception as e:
            self.logger.error(f"Error in sympy_to_matlab conversion: {e}", exc_info=True)
            raise

    def _handle_list_expression(self, expr):
        """Handle list expressions."""
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
        """Handle integral expressions."""
        self.logger.debug(f"Handling integral expression: {expr}")
        try:
            func = self.sympy_to_matlab(expr.function)
            var = expr.variables[0]
            
            self.logger.debug(f"Integral function: {func}, variable: {var}")
            
            # Handle logarithm conversions
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
        """Handle derivative expressions."""
        self.logger.debug(f"Handling derivative expression: {expr}")
        try:
            # Get the function being differentiated
            func = self.sympy_to_matlab(expr.expr)
            
            # Get the variable and order of differentiation
            var = str(expr.variables[0])
            
            # Handle order of derivative
            if hasattr(expr, 'order'):
                order = expr.order
            else:
                # Count the number of times the same variable appears
                order = sum(1 for v in expr.variables if str(v) == var)
            
            # Convert logarithms if present
            func = self._convert_logarithms(func)
            
            # Create MATLAB derivative command
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
        """Process variables for derivative expressions."""
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
        """Handle equation expressions."""
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
        """Handle function expressions."""
        self.logger.debug(f"Handling function expression: {expr}")
        try:
            func_name = expr.func.__name__
            args = [self.sympy_to_matlab(arg) for arg in expr.args]
            
            function_handlers = {
                'sin': lambda args: f"sind({args[0]})" if self.combo_angle.currentText() == 'Degree' else f"sin({args[0]})",
                'cos': lambda args: f"cosd({args[0]})" if self.combo_angle.currentText() == 'Degree' else f"cos({args[0]})",
                'tan': lambda args: f"tand({args[0]})" if self.combo_angle.currentText() == 'Degree' else f"tan({args[0]})",
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
        """Process and convert expression string to MATLAB format."""
        expr_str = str(expr)
        
        # Convert logarithms
        expr_str = self._convert_logarithms(expr_str)
        
        # Apply function replacements
        expr_str = self._apply_function_replacements(expr_str)
        
        # Handle integrals and derivatives
        expr_str = self._handle_integral_derivative_patterns(expr_str)
        
        # Process operators and special cases
        expr_str = self._process_operators(expr_str)
        
        # Handle combinations and permutations
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
        """Apply function replacements using the replacement dictionary."""
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
        """Handle integral and derivative patterns."""
        patterns = {
            r'Integral\((.*?), \((.*?), (.*?), (.*?)\)\)': r'integral(@(x) \1, \3, \4)',
            r'Integral\((.*?), (.*?)\)': r'int(\1, \2)',
            r'Function\((.*?)\)': r'\1'
        }
        
        for pattern, replacement in patterns.items():
            expr_str = re.sub(pattern, replacement, expr_str)
        return expr_str

    def _process_operators(self, expr_str):
        """Process mathematical operators."""
        expr_str = expr_str.replace('**', '.^')
        expr_str = expr_str.replace('/', './').replace('*', '.*')
        expr_str = expr_str.replace('.*1', '').replace('.*0', '0')
        return expr_str

    def _handle_combinations_permutations(self, expr_str):
        """Handle combinations and permutations patterns."""
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
        input_mode = self.combo_mode.currentText()
        angle_mode = self.combo_angle.currentText()

        try:
            if input_mode == 'Matrix':
                self.handle_matrix_calculation()
            elif input_mode == 'LaTeX':
                self.handle_latex_calculation(angle_mode)
            elif input_mode == 'MATLAB':
                self.handle_matlab_calculation()
            else:
                raise ValueError("Unsupported input mode.")
        except matlab.engine.MatlabExecutionError as me:
            QMessageBox.critical(self, "MATLAB Error", f"MATLAB Error: {me}")
        except ValueError as ve:
            QMessageBox.critical(self, "Value Error", f"Value Error: {ve}")
        except Exception as e:
            QMessageBox.critical(self, "Unexpected Error", f"Unexpected Error: {str(e)}")
    
    def format_matrix_result(self, result, operation):
        if result is None:
            return "Operation Failed."

        if isinstance(result, list):
            # For matrix/list results, format as a string representation
            return str(result)
        else:
            # For scalar results like Determinant or Rank
            return str(result)

    def handle_matrix_calculation(self):
        matrix_text = self.matrix_input.toPlainText().strip()
        operation = self.combo_matrix_op.currentText()
        
        if not matrix_text:
            QMessageBox.critical(self, "Input Error", "Please enter a matrix.")
            return
        
        # Check if the input is a stored matrix name
        if matrix_text in self.matrix_memory:
            matrix_text = self.matrix_memory[matrix_text]

        # Evaluate the matrix in MATLAB
        try:
            self.eng.eval(f"matrix = {matrix_text};", nargout=0)
        except matlab.engine.MatlabExecutionError as me:
            QMessageBox.critical(self, "MATLAB Error", f"Error evaluating matrix: {me}")
            return
        
        try:
            if operation == 'Determinant':
                result = self.eng.eval("det(matrix)", nargout=1)
                result = round(float(result), 2)  # Round to 2 decimal places

            elif operation == 'Inverse':
                result = self.eng.eval("inv(matrix)", nargout=1)
                result = np.round(np.array(result), 2).tolist()  # Round each element to 2 decimal places

            elif operation == 'Eigenvalues':
                result = self.eng.eval("eig(matrix)", nargout=1)
                result = np.round(np.array(result), 2).tolist()  # Round each eigenvalue to 2 decimal places

            elif operation == 'Rank':
                result = self.eng.eval("rank(matrix)", nargout=1)
                result = int(result)  # Rank is always an integer, no rounding needed

            elif operation in ['Multiply', 'Add', 'Subtract', 'Divide']:
                result = self.handle_matrix_arithmetic(operation)

            elif operation == 'Differentiate':
                result = self.handle_matrix_differentiation()

            else:
                QMessageBox.warning(self, "Operation Warning", f"Operation '{operation}' is not supported.")
                return

            # Convert result to string for display
            result = self.format_matrix_result(result, operation)

            # Set the result label text and font
            self.result_label.setText(f"Result: {result}")
            self.result_label.setFont(QFont("Arial", 13, QFont.Bold))

        except matlab.engine.MatlabExecutionError as me:
            QMessageBox.critical(self, "MATLAB Error", f"MATLAB Error: {me}")

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

            # Parse the LaTeX-like expression to SymPy
            sympy_expr = parse_latex_expression(expression)
            self.logger.debug(f"Converted to SymPy expression: '{sympy_expr}'")

            # Convert the SymPy expression to MATLAB
            sympy_to_matlab_converter = SympyToMatlab()
            matlab_expression = sympy_to_matlab_converter.sympy_to_matlab(sympy_expr)
            self.logger.debug(f"Converted to MATLAB expression: '{matlab_expression}'")

            # Handle trigonometric functions based on angle mode
            if angle_mode == 'Degree':
                # Convert trig functions to degree versions
                matlab_expression = re.sub(r'\bsin\((.*?)\)', lambda m: f"sind({m.group(1)})", matlab_expression)
                matlab_expression = re.sub(r'\bcos\((.*?)\)', lambda m: f"cosd({m.group(1)})", matlab_expression)
                matlab_expression = re.sub(r'\btan\((.*?)\)', lambda m: f"tand({m.group(1)})", matlab_expression)

            # Evaluate the expression in MATLAB
            self.eng.eval(f"temp_result = {matlab_expression};", nargout=0)
            
            # Check if result is symbolic
            is_symbolic = self.eng.eval("isa(temp_result, 'sym')", nargout=1)
            
            if is_symbolic:
                # For symbolic results, convert to string
                displayed_result = self.eng.eval("char(temp_result)", nargout=1)
            else:
                # For numeric results, convert to double
                result = self.eng.eval("double(temp_result)", nargout=1)
                if isinstance(result, float):
                    displayed_result = f"{result:.6g}"
                else:
                    displayed_result = str(result)

            # Post-process the result string
            displayed_result = displayed_result.replace('log(', 'ln(')
            displayed_result = displayed_result.replace('.*', '*')
            displayed_result = displayed_result.replace('.^', '^')
            displayed_result = displayed_result.replace('cosd(', 'cos(')
            displayed_result = displayed_result.replace('sind(', 'sin(')
            displayed_result = displayed_result.replace('tand(', 'tan(')

            self.result_display.setText(displayed_result)
            self.logger.debug(f"Displayed Result: {displayed_result}")
            
        except Exception as e:
            self.logger.error(f"Error in LaTeX calculation: {e}")
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def evaluate_expression(self, matlab_expression):
        """Evaluate the expression using the MATLAB engine."""
        try:
            # For limit expressions, add numerical evaluation
            if matlab_expression.startswith('limit('):
                # First calculate the symbolic limit
                self.eng.eval(f"temp_result = {matlab_expression};", nargout=0)
                # Then convert to double for numerical evaluation
                self.eng.eval("temp_result = double(temp_result);", nargout=0)
            else:
                self.eng.eval(f"temp_result = {matlab_expression};", nargout=0)
            
            # Get the result
            result = self.eng.eval("temp_result", nargout=1)
            
            # If the result is a float, round it to a reasonable number of decimal places
            if isinstance(result, float):
                result = round(result, 6)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in evaluate_expression: {e}")
            raise

    def _is_numeric_expression(self, sympy_expr):
        """Determine if the SymPy expression is numeric."""
        return sympy_expr.is_number

    def _extract_function_argument(self, expression):
        """Extract the function name and its argument."""
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
            # Preprocess the expression
            processed_expr = EvaluateExpression.preprocess_expression(expression)
            self.logger.debug(f"Processed expression for SymPy: {processed_expr}")
            
            # Sympify the transformed expression
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
            # Handle trigonometric functions based on angle mode
            if self.combo_angle.currentText() == 'Degree':
                # Convert trig functions to degree versions
                input_text = re.sub(r'\bsin\((.*?)\)', lambda m: f"sind({m.group(1)})", input_text)
                input_text = re.sub(r'\bcos\((.*?)\)', lambda m: f"cosd({m.group(1)})", input_text)
                input_text = re.sub(r'\btan\((.*?)\)', lambda m: f"tand({m.group(1)})", input_text)

            # Convert ln to log for MATLAB processing
            input_text = input_text.replace('ln(', 'log(')

            # Assign the expression to a MATLAB variable
            self.eng.eval(f"result = {input_text};", nargout=0)

            # Evaluate the expression
            result = self.eng.eval("result", nargout=1)

            # Convert symbolic result to string if necessary
            if isinstance(result, matlab.object):
                result = self.eng.eval("char(result)", nargout=1)
            elif isinstance(result, (int, float)):
                result = f"{result:.4f}"
            else:
                result = str(result)

            # Replace 'inf' with the infinity symbol '∞'
            if 'inf' in result:
                result = result.replace('inf', '∞')

            # Post-process the result: replace all instances of log with ln
            result = result.replace('log(', 'ln(')
            result = result.replace('==', '=')

            self.result_label.setText(f"Result: {result}")
            self.result_label.setFont(QFont("Monaspace Neon", 14))

        except matlab.engine.MatlabExecutionError as me:
            QMessageBox.critical(self, "MATLAB Error", f"MATLAB Error: {me}")
        except Exception as e:
            QMessageBox.critical(self, "Unexpected Error", f"Unexpected Error: {str(e)}")
    
    def closeEvent(self, event):
        """Handle the application closing event."""
        try:
            self.eng.quit()
            self.logger.info("MATLAB engine terminated")
        except Exception as e:
            self.logger.warning(f"Error terminating MATLAB engine: {e}")
        event.accept()

    def calculate_matrix(self):
        """Handle the calculation for matrix operations."""
        try:
            # Retrieve the matrix text and selected operation
            matrix_text = self.matrix_input.toPlainText().strip()
            operation = self.combo_matrix_op.currentText()
            
            if not matrix_text:
                QMessageBox.warning(self, "Input Error", "Please enter a matrix.")
                return
            
            # Check if the input is a stored matrix name
            if matrix_text in self.matrix_memory:
                matrix_text = self.matrix_memory[matrix_text]

            # Evaluate the matrix in MATLAB
            self.eng.eval(f"matrix = {matrix_text};", nargout=0)
            
            # Perform the selected operation
            if operation == 'Determinant':
                result = self.eng.eval("det(matrix)", nargout=1)
                result = round(float(result), 2)  # Round to 2 decimal places

            elif operation == 'Inverse':
                result = self.eng.eval("inv(matrix)", nargout=1)
                result = np.round(np.array(result), 2).tolist()  # Round each element to 2 decimal places

            elif operation == 'Eigenvalues':
                result = self.eng.eval("eig(matrix)", nargout=1)
                result = np.round(np.array(result), 2).tolist()  # Round each eigenvalue to 2 decimal places

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

            # Convert result to string for display
            result_str = self.format_matrix_result(result, operation)

            # Set the result in the result_display text area
            self.result_display.setText(result_str)
            self.result_display.setFont(QFont("Monaspace Neon", 14))

        except matlab.engine.MatlabExecutionError as me:
            QMessageBox.critical(self, "MATLAB Error", f"MATLAB Error: {me}")
        except Exception as e:
            QMessageBox.critical(self, "Unexpected Error", f"Unexpected Error: {str(e)}")

if __name__ == '__main__':
    download_fonts()
    app = QApplication(sys.argv)
    calculator = CalculatorApp()
    calculator.show()
    sys.exit(app.exec_())
