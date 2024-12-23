import sys
import os
import re
import logging
from themes.theme_manager import ThemeManager, get_tokyo_night_theme, get_aura_theme, get_light_theme
from ui.legend_window import LegendWindow
from ui.ui_config import UiConfig
from modules.modules_import import PackageImporter
from latex_pack.shortcut import ExpressionShortcuts
from matlab_interface.sympy_to_matlab import SympyToMatlab

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


class CalculatorApp(QWidget):
    """
    Main calculator application window with support for LaTeX, MATLAB, and Matrix operations.
    """

    # Class constants
    FORMULA_FONT = QFont("Arial", 13, QFont.Bold)
    PLACEHOLDER_TEXT = (
        'Enter LaTeX expression, e.g., {latex_example}\n'
        'Or MATLAB expression, e.g., {matlab_example}'
    ).format(
        latex_example=r'\binom{5}{2} + \sin\left(\frac{\pi}{2}\right)',
        matlab_example='nchoosek(5,2) + sin(pi/2)'
    )

    def __init__(self):
        """Initialize the calculator application."""
        super().__init__()
        
        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('calculator.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Calculator Application")
        
        self.theme_manager = ThemeManager()
        self.legend_window = None
        self.init_ui()
        self._init_theme()
        self.matrix_memory = {}
        
        try:
            self.eng = matlab.engine.start_matlab()
            self.eng.eval("syms x f(x)", nargout=0)
            self.logger.info("MATLAB engine started successfully")
        except Exception as e:
            self.logger.error(f"Failed to start MATLAB engine: {e}")

        self.sympy_converter = SympyToMatlab()

    def _init_theme(self):
        """Initialize and set default theme."""
        self.set_theme("aura")

    def init_ui(self):
        """Initialize the calculator's user interface."""
        self._setup_window()
        main_layout = QVBoxLayout()
        
        # Initialize all UI components using helper methods
        layouts = {
            'top': self._create_top_buttons(),
            'mode': self._create_mode_selection(),
            'angle': self._create_angle_selection(),
            'matrix': self._create_matrix_components(),
            'formula': self._create_formula_components(),
            'result': self._create_result_components()
        }
        
        # Add all layouts to main layout in order
        for layout in layouts.values():
            if isinstance(layout, (list, tuple)):
                for item in layout:
                    main_layout.addLayout(item) if isinstance(item, QHBoxLayout) else main_layout.addWidget(item)
            else:
                main_layout.addLayout(layout) if isinstance(layout, QHBoxLayout) else main_layout.addWidget(layout)
        
        self.setLayout(main_layout)

    def _setup_window(self):
        """Set up the main window properties."""
        self.setWindowTitle('Scientific Calculator')
        self.setGeometry(100, 100, 600, 450)

    def _create_top_buttons(self):
        """Create theme and legend buttons."""
        top_layout = QHBoxLayout()
        
        # Create UiConfig instance and get configurations
        ui_config = UiConfig()
        ui_config.config_button(
            theme_callback=self.show_theme_menu,
            legend_callback=self.show_legend
        )
        
        top_layout.addStretch()
        for btn_name, config in ui_config.button_configs.items():
            button = QPushButton(config['text'])
            button.setFixedSize(*config['size'])
            button.setStyleSheet(ui_config.button_style)
            button.clicked.connect(config['callback'])
            setattr(self, f"{btn_name}_button", button)
            top_layout.addWidget(button)
        
        return top_layout

    def _create_mode_selection(self):
        """Create input mode selection components."""
        mode_layout = QHBoxLayout()
        self.label_mode = QLabel('Input Mode:')
        self.label_mode.setFixedWidth(100)
        
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(['LaTeX', 'MATLAB', 'Matrix'])
        self.combo_mode.setFixedWidth(100)
        self.combo_mode.currentTextChanged.connect(self.on_mode_changed)
        
        mode_layout.addWidget(self.label_mode)
        mode_layout.addWidget(self.combo_mode)
        mode_layout.addStretch()
        return mode_layout

    def _create_angle_selection(self):
        """Create angle mode selection components."""
        angle_layout = QHBoxLayout()
        self.label_angle = QLabel('Angle Mode:')
        self.label_angle.setFixedWidth(100)
        
        self.combo_angle = QComboBox()
        self.combo_angle.addItems(['Degree', 'Radian'])
        self.combo_angle.setFixedWidth(100)
        
        angle_layout.addWidget(self.label_angle)
        angle_layout.addWidget(self.combo_angle)
        angle_layout.addStretch()
        return angle_layout

    def _create_matrix_components(self):
        """Create matrix-related UI components."""
        # Matrix input
        self.matrix_input = QTextEdit()
        self.matrix_input.setFixedHeight(100)
        self.matrix_input.setPlaceholderText(
            "Enter matrix in MATLAB format, e.g., [1 2; 3 4]\n"
            "Or [1, 2; 3, 4] for comma-separated values"
        )
        self.matrix_input.hide()
        
        # Matrix operations
        matrix_op_layout = QHBoxLayout()
        self.label_matrix_op = QLabel('Matrix Operation:')
        self.combo_matrix_op = QComboBox()
        self.combo_matrix_op.addItems([
            'Determinant', 'Inverse', 'Eigenvalues', 'Rank',
            'Multiply', 'Add', 'Subtract', 'Divide', 'Differentiate'
        ])
        self.combo_matrix_op.setFixedWidth(130)
        
        matrix_op_layout.addWidget(self.label_matrix_op)
        matrix_op_layout.addWidget(self.combo_matrix_op)
        matrix_op_layout.addStretch()
        
        # Matrix memory buttons
        matrix_memory_layout = QHBoxLayout()
        self.store_matrix_button = QPushButton('Store Matrix')
        self.recall_matrix_button = QPushButton('Recall Matrix')
        
        self.store_matrix_button.clicked.connect(self.store_matrix)
        self.recall_matrix_button.clicked.connect(self.recall_matrix)
        
        matrix_memory_layout.addWidget(self.store_matrix_button)
        matrix_memory_layout.addWidget(self.recall_matrix_button)
        
        # Hide matrix-related components initially
        for widget in [self.label_matrix_op, self.combo_matrix_op,
                      self.store_matrix_button, self.recall_matrix_button]:
            widget.hide()
        
        return [self.matrix_input, matrix_op_layout, matrix_memory_layout]

    def _create_formula_components(self):
        """Create formula input components."""
        components = []
        
        # Create label
        self.label_formula = QLabel('Math Expression:')
        self.label_formula.setFont(self.FORMULA_FONT)
        components.append(self.label_formula)
        
        # Create text entry
        self.entry_formula = QTextEdit()
        self.entry_formula.setFixedHeight(100)
        self.entry_formula.setPlaceholderText(self.PLACEHOLDER_TEXT)
        components.append(self.entry_formula)
        
        # Create calculate button
        self.calculate_button = QPushButton('Calculate')
        self.calculate_button.clicked.connect(self.calculate)
        components.append(self.calculate_button)
        
        return components

    def _create_result_components(self):
        """Create result display components."""
        self.result_label = QLabel('Result: ')
        self.result_label.setFont(QFont("Arial", 13, QFont.Bold))
        return self.result_label

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
                'nchoosek': lambda args: f"nchoosek({args[0]}, {args[1]})"
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

    def _convert_logarithms(self, expr_str):
        """Convert different types of logarithms."""
        if 'log(x, E)' in expr_str or 'log(x)' in expr_str:
            expr_str = expr_str.replace('log(x, E)', 'ln(x)')
            expr_str = expr_str.replace('log(x)', 'ln(x)')
        elif 'log(x, 10)' in expr_str:
            expr_str = expr_str.replace('log(x, 10)', 'log10(x)')
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
    
    def _convert_logarithms(self, expr_str):
        """Convert different types of logarithms."""
        # Don't convert to ln() - keep as log() for MATLAB compatibility
        if 'log(x, E)' in expr_str or 'log(x)' in expr_str:
            expr_str = expr_str.replace('log(x, E)', 'log(x)')
        elif 'log(x, 10)' in expr_str:
            expr_str = expr_str.replace('log(x, 10)', 'log10(x)')
        return expr_str

    def handle_latex_calculation(self, angle_mode):
        input_text = self.entry_formula.toPlainText().strip()
        
        if not input_text:
            QMessageBox.warning(self, "Input Error", "Please enter a mathematical expression.")
            return

        try:
            # Initialize MATLAB symbolic variables
            self.eng.eval("syms x y z t", nargout=0)

            # First, identify the type of expression
            is_derivative = any(key in input_text for key in ExpressionShortcuts.DERIVATIVE_SHORTCUTS)
            is_integral = any(key in input_text for key in ExpressionShortcuts.INTEGRAL_SHORTCUTS)
            is_function = any(key in input_text for key in ExpressionShortcuts.FUNCTION_SHORTCUTS)

            # Pre-process function names before any other operations
            if is_function:
                # Create a list of functions sorted by length (longest first) to avoid partial matches
                functions = sorted(ExpressionShortcuts.FUNCTION_SHORTCUTS.keys(), key=len, reverse=True)
                for func in functions:
                    if func in input_text.lower():
                        # Handle degree mode for trigonometric functions
                        if func in ['sin', 'cos', 'tan'] and angle_mode == 'Degree':
                            input_text = input_text.lower().replace(func, func + 'd')
                        else:
                            input_text = input_text.lower().replace(func, func)

            # Convert based on expression type
            if is_derivative:
                # Handle derivative expressions directly
                if 'd/dx' in input_text:
                    parts = input_text.split('d/dx')
                    if len(parts) == 2:
                        expr = parts[1].strip()
                        expr = self._add_multiplication_operators(expr)
                        matlab_expression = f"diff({expr}, x)"
                elif 'd/dy' in input_text:
                    parts = input_text.split('d/dy')
                    if len(parts) == 2:
                        expr = parts[1].strip()
                        expr = self._add_multiplication_operators(expr)
                        matlab_expression = f"diff({expr}, y)"
                else:
                    # For higher order derivatives
                    match = re.match(r'd(\d*)/d([xyz])(\d*)\s*(.*)', input_text)
                    if match:
                        order, var, _, expr = match.groups()
                        order = order if order else '1'
                        expr = self._add_multiplication_operators(expr)
                        matlab_expression = f"diff({expr}, {var}, {order})"
                    else:
                        raise ValueError("Invalid derivative expression")
            
            elif is_integral:
                # Handle integral expressions directly
                if 'dx' in input_text:
                    # Remove 'int' or '\int' and split by 'dx'
                    expr = input_text.replace('int', '').replace('\\int', '').split('dx')[0].strip()
                    expr = self._add_multiplication_operators(expr)
                    matlab_expression = f"int({expr}, x)"
                elif 'dy' in input_text:
                    expr = input_text.replace('int', '').replace('\\int', '').split('dy')[0].strip()
                    expr = self._add_multiplication_operators(expr)
                    matlab_expression = f"int({expr}, y)"
                else:
                    expr = input_text.replace('int', '').replace('\\int', '').strip()
                    expr = self._add_multiplication_operators(expr)
                    matlab_expression = f"int({expr}, x)"
            
            elif is_function:
                # Remove spaces between function name and parentheses
                matlab_expression = re.sub(r'(\w+)\s+\(', r'\1(', input_text)
                # Add multiplication operators where needed, but skip function names
                matlab_expression = self._add_multiplication_operators(matlab_expression, skip_functions=True)
            
            else:
                # For basic expressions
                matlab_expression = self._add_multiplication_operators(input_text)

            self.logger.debug(f"Final MATLAB Expression: {matlab_expression}")

            # Evaluate the expression
            try:
                self.eng.eval(f"temp_result = {matlab_expression};", nargout=0)
                if matlab_expression.startswith(('diff(', 'int(')):
                    self.eng.eval("temp_result = simplify(temp_result);", nargout=0)
                
                # Get the result
                result_str = self.eng.eval("char(temp_result)", nargout=1)
                result = result_str.replace('.^', '^').replace('.*', '*')
                
                self.result_label.setText(f"Result: {result}")
                self.result_label.setFont(QFont("Arial", 13, QFont.Bold))
                self.logger.debug(f"Result: {result}")

            except matlab.engine.MatlabExecutionError as me:
                raise ValueError(f"MATLAB Error: {me}")

            # Clean up MATLAB workspace
            self.eng.eval("clear temp_result", nargout=0)

        except Exception as e:
            self.logger.error(f"Error evaluating expression: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Error evaluating expression: {str(e)}")

    def _add_multiplication_operators(self, expr, skip_functions=False):
        """
        Add multiplication operators where needed.
        
        Args:
            expr (str): Expression to process
            skip_functions (bool): If True, don't add operators within function names
        """
        if skip_functions:
            # First, temporarily replace function names with placeholders
            functions = sorted(ExpressionShortcuts.FUNCTION_SHORTCUTS.keys(), key=len, reverse=True)
            replacements = {}
            for i, func in enumerate(functions):
                if func in expr.lower():
                    placeholder = f"__FUNC{i}__"
                    replacements[placeholder] = func
                    expr = expr.lower().replace(func, placeholder)

        # Add * between number and variable
        expr = re.sub(r'(\d+)([a-zA-Z])', r'\1.*\2', expr)
        # Add * between closing parenthesis and number
        expr = re.sub(r'\)(\d+)', r')*\1', expr)
        # Add * between variable and number
        expr = re.sub(r'([a-zA-Z])(\d+)', r'\1.*\2', expr)
        # Add * between variable and variable (but not in function names)
        if not skip_functions:
            expr = re.sub(r'([a-zA-Z])([a-zA-Z])', r'\1.*\2', expr)
        
        if skip_functions:
            # Restore function names
            for placeholder, func in replacements.items():
                expr = expr.replace(placeholder, func)

        # Remove spaces between function name and parentheses
        expr = re.sub(r'(\w+)\s+\(', r'\1(', expr)
        # Remove any double .* that might have been created
        expr = expr.replace('.*.*', '.*')
        return expr

    def handle_regular_expression(self, matlab_expression):
        """Handle non-derivative expressions."""
        try:
            # Expression is already cleaned up by _prepare_matlab_expression
            result = self.eng.eval(f"simplify({matlab_expression})", nargout=1)
            
            # Format the result
            if isinstance(result, matlab.double):
                if len(result) == 1:
                    return f"{float(result):.4f}"
                return str(list(result))
            elif isinstance(result, matlab.object):
                result_str = self.eng.eval("char(simplify(result))", nargout=1)
                return result_str.replace('.^', '^').replace('.*', '*')
            else:
                return str(result)
                
        except matlab.engine.MatlabExecutionError as me:
            raise ValueError(f"MATLAB Error: {me}")

    def handle_integral(self, matlab_expression):
        try:
            # Convert any remaining 'ln' to 'log' before sending to MATLAB
            matlab_expression = matlab_expression.replace('ln(', 'log(')
            
            # Assign the integral to a MATLAB variable
            self.eng.eval(f"result = {matlab_expression};", nargout=0)

            # Simplify the result and convert to string
            self.eng.eval("result = simplify(result);", nargout=0)
            
            # Get the result as a string and replace all instances of log with ln
            result_str = self.eng.eval("char(result)", nargout=1)
            result = result_str.replace('log(', 'ln(')
            
            return result

        except matlab.engine.MatlabExecutionError as me:
            QMessageBox.critical(self, "MATLAB Error", f"MATLAB Error: {me}")
            return "Error in integration."

    def handle_derivative(self, matlab_expression, angle_mode):
        try:
            matlab_expression = matlab_expression.replace('ln(', 'log(')
            
            # Assign the derivative to a MATLAB variable
            self.eng.eval(f"result = {matlab_expression};", nargout=0)

            # Simplify the result
            self.eng.eval("result = simplify(result);", nargout=0)
            result = self.eng.eval("result", nargout=1)

            # Convert symbolic result to string
            if isinstance(result, matlab.object):
                result_str = self.eng.eval("char(result)", nargout=1)
                result = result_str
            else:
                result = str(result)
            
            result = result.replace('log(', 'ln(')

            return result

        except matlab.engine.MatlabExecutionError as me:
            QMessageBox.critical(self, "MATLAB Error", f"MATLAB Error: {me}")
            return "Error in differentiation."

    def handle_equation(self, matlab_expression, angle_mode):
        try:
            # Determine if it's a trigonometric equation
            is_trig = any(func in matlab_expression for func in ['sin', 'cos', 'tan', 'sind', 'cosd', 'tand'])
            is_equation = '==' in matlab_expression

            if is_trig and is_equation:
                if angle_mode.lower() == 'degree':
                    # Replace trig functions with degree counterparts if not already done
                    matlab_expression = re.sub(r'\bsin\(', 'sind(', matlab_expression, flags=re.IGNORECASE)
                    matlab_expression = re.sub(r'\bcos\(', 'cosd(', matlab_expression, flags=re.IGNORECASE)
                    matlab_expression = re.sub(r'\btan\(', 'tand(', matlab_expression, flags=re.IGNORECASE)
                    solve_command = f"vpasolve({matlab_expression}, x, [0, 360])"
                
                    # Get the result
                    result = self.eng.eval(solve_command, nargout=1)
                
                    # Convert result to degrees
                    if isinstance(result, matlab.double):
                        result = np.array(result).flatten().tolist()
                        # Round to 2 decimal places
                        result = [f"{x:.2f}°" for x in result]
                        result = f"x = {', '.join(result)}"
                    else:
                        result_str = self.eng.eval(f"char({solve_command})", nargout=1)
                        result = f"x = {result_str}°"

                else:
                    # For radian mode
                    solve_command = f"vpasolve({matlab_expression}, x, [0, 2*pi])"
                    result = self.eng.eval(solve_command, nargout=1)
                    
                    if isinstance(result, matlab.double):
                        result = np.array(result).flatten().tolist()
                        result = [f"{x/np.pi:.4f}π" if x != 0 else "0" for x in result]
                        result = f"x = {', '.join(result)}"
                    else:
                        result_str = self.eng.eval(f"char({solve_command})", nargout=1)
                        result = f"x = {result_str} rad"

                # Handle case when no solution is found
                if result in ["x = []", "x = []°", "x = [] rad"]:
                    result = "No real solutions found"
            
                return result

            else:
                # Handle non-trigonometric equations
                solve_command = f"solve({matlab_expression}, x)"
                result = self.eng.eval(solve_command, nargout=1)

                # Convert result to a more readable format
                if isinstance(result, matlab.double):
                    result = np.array(result).flatten().tolist()
                    result = [f"{x:.4f}" for x in result]
                    result = f"x = {', '.join(result)}"
                elif isinstance(result, matlab.object):
                    result_str = self.eng.eval(f"char(solve({matlab_expression}, x))", nargout=1)
                    result_str = result_str.replace('==', '=')
                    result = f"x = {result_str}"
                else:
                    result = str(result)

                return result

        except matlab.engine.MatlabExecutionError as me:
            QMessageBox.critical(self, "MATLAB Error", f"MATLAB Error: {me}")
            return "Error in solving equation."

    def handle_regular_expression(self, matlab_expression):
        try:
            # Assign the expression to a MATLAB variable
            self.eng.eval(f"result = {matlab_expression};", nargout=0)

            # Evaluate the expression
            result = self.eng.eval("result", nargout=1)

            # Convert symbolic result to string if necessary
            if isinstance(result, matlab.object):
                result_str = self.eng.eval("char(result)", nargout=1)
                # Remove the "Symbolic expression:" prefix
                result = result_str.strip()
            elif isinstance(result, (int, float)):
                result = f"{result:.4f}"
            else:
                result = str(result)
            
            result = result.replace('log(', 'ln(')

            return result

        except matlab.engine.MatlabExecutionError as me:
            QMessageBox.critical(self, "MATLAB Error", f"MATLAB Error: {me}")
            return "Error in evaluating expression."

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
                result_str = self.eng.eval("char(result)", nargout=1)
                result = result_str
            elif isinstance(result, (int, float)):
                result = f"{result:.4f}"
            else:
                result = str(result)

            # Post-process the result: replace all instances of log with ln
            result = result.replace('log(', 'ln(')
            result = result.replace('==', '=')

            self.result_label.setText(f"Result: {result}")
            self.result_label.setFont(QFont("Arial", 13, QFont.Bold))

        except matlab.engine.MatlabExecutionError as me:
            QMessageBox.critical(self, "MATLAB Error", f"MATLAB Error: {me}")
        except Exception as e:
            QMessageBox.critical(self, "Unexpected Error", f"Unexpected Error: {str(e)}")
    
    def closeEvent(self, event):
        self.eng.quit()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    calculator = CalculatorApp()
    calculator.show()
    sys.exit(app.exec_())
