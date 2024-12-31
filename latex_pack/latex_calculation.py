from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QFont
import re
import matlab.engine
from matlab_interface.evaluate_expression import EvaluateExpression
from matlab_interface.display import Display
import logging

class LatexCalculation:
    # Define supported functions, ordered from longest to shortest to prevent partial matches
    SUPPORTED_FUNCTIONS = [
        'log10', 'ln', 'asin', 'acos', 'atan', 'cot', 'sec', 'csc',
        'sin', 'cos', 'tan', 'log', 'sqrt', 'abs', 'exp', 'sum', 'prod'
    ]

    def __init__(self):
        """Initialize the LatexCalculation class."""
        self.logger = logging.getLogger(__name__)
        self._configure_logger()
        self.display = None  # Placeholder for Display instance

    def setup(self, eng, entry_formula, result_label):
        """Set up the calculator with required components."""
        self.eng = eng
        self.entry_formula = entry_formula
        self.result_label = result_label
        self.display = Display(self.result_label)  # Initialize Display
        return self

    def _configure_logger(self):
        """Configure the logger for the LatexCalculation class."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)

    def _handle_latex_calculation(self, angle_mode):
        try:
            input_text = self._validate_and_prepare_input()
            matlab_expression = self._process_expression(input_text, angle_mode)
            result = self._evaluate_and_format_result(matlab_expression)
            self.display.display_result(result)  # Use Display class
        except Exception as e:
            self._handle_error(e)
        finally:
            self._cleanup_matlab_workspace()

    def _validate_and_prepare_input(self):
        input_text = self.entry_formula.toPlainText().strip()
        if not input_text:
            raise ValueError("Please enter a mathematical expression.")
            
        # Preprocess and normalize
        input_text = self._preprocess_latex_functions(input_text)
        return re.sub(r'\s+', ' ', input_text).strip()

    def _preprocess_latex_functions(self, expr):
        """Preprocess LaTeX function notation to MATLAB notation."""
        self.logger.debug(f"Initial expression: {expr}")

        # Enhanced sum pattern with case-insensitivity and better space handling
        sum_pattern = r'(?i)sum\s*\(\s*(\d+)\s*to\s*(\d+)\s*\)\s*([a-zA-Z_]\w*)'
        def replace_sum(match):
            start, end, var = match.groups()
            return f"symsum({var}^2, {var}, {start}, {end})"

        expr = re.sub(sum_pattern, replace_sum, expr)
        self.logger.debug(f"After sum conversion: {expr}")

        # Handle operator shortcuts
        operator_shortcuts = {
            r'prod\s*\((\d+)\s*to\s*(\d+)\)\s*([a-zA-Z0-9]+)': r'\\prod_{\1}^{\2} \3'
        }

        for shortcut_pattern, latex_pattern in operator_shortcuts.items():
            expr = re.sub(shortcut_pattern, latex_pattern, expr)
            self.logger.debug(f"Converted shortcut to LaTeX: {expr}")

        # Dictionary of common LaTeX functions and their MATLAB equivalents
        latex_funcs = {
            r'\\sin': 'sin',
            r'\\cos': 'cos',
            r'\\tan': 'tan',
            r'\\cot': 'cot',
            r'\\sec': 'sec',
            r'\\csc': 'csc',
            r'\\asin': 'asin',
            r'\\acos': 'acos',
            r'\\atan': 'atan',
            r'\\ln': 'ln',       # Keep ln as ln
            r'\\log': 'log10',   # LaTeX \log -> MATLAB log10
            r'\\sqrt': 'sqrt',
            r'\\abs': 'abs',
            r'\\exp': 'exp',
            r'\\sum': 'symsum',
            r'\\prod': 'prod'
        }

        # Replace LaTeX function notation with MATLAB notation
        for latex_func, matlab_func in latex_funcs.items():
            expr = re.sub(latex_func + r'\s*{([^}]+)}', rf'{matlab_func}(\1)', expr)
            expr = re.sub(latex_func + r'\s*\(([^)]+)\)', rf'{matlab_func}(\1)', expr)
            expr = re.sub(latex_func + r'\s+([a-zA-Z0-9]+)', rf'{matlab_func}(\1)', expr)
            self.logger.debug(f"Converted LaTeX to MATLAB: {expr}")

        self.logger.debug(f"Final preprocessed expression: {expr}")
        return expr

    def _process_expression(self, expr, angle_mode):
        """Process the input expression to a valid MATLAB expression."""
        self.logger.debug(f"Processing expression: {expr}")
        
        # Handle derivatives
        if expr.startswith('d/dx'):
            return self._process_derivative(expr)
        
        # Handle integrals
        if expr.startswith('int'):
            return self._process_integral(expr)
        
        if expr.startswith('sum'):
            return self._process_sum(expr)
        
        if expr.startswith('prod'):
            return self._process_product(expr)
        
        # Add multiplication operators where necessary
        expr = self._add_multiplication_operators(expr)
        
        # Convert trigonometric functions to handle degrees if needed
        if angle_mode.lower() == 'degree':
            expr = self._convert_trig_to_degrees(expr)
        
        self.logger.debug(f"Final MATLAB Expression: {expr}")
        return expr

    def _process_integral(self, expr):
        """Process the integral expression."""
        self.logger.debug(f"Processing integral: {expr}")
        # Expected format: int expression dx
        match = re.match(r'int\s+(.*?)\s+dx', expr)
        if not match:
            raise ValueError("Invalid integral format. Use 'int expression dx'.")
        integrand = match.group(1)
        matlab_expr = f'int({integrand}, x)'
        self.logger.debug(f"Converted integral to MATLAB expression: '{matlab_expr}'")
        return matlab_expr

    def _process_derivative(self, expr):
        """Process the derivative expression."""
        self.logger.debug(f"Processing derivative: {expr}")
        # Expected format: d/dx expression
        match = re.match(r'd/dx\s+(.*)', expr)
        if not match:
            raise ValueError("Invalid derivative format. Use 'd/dx expression'.")
        derivative_expr = match.group(1)
        matlab_expr = f'diff({derivative_expr}, x)'
        self.logger.debug(f"Converted derivative to MATLAB expression: '{matlab_expr}'")
        return matlab_expr

    def _process_sum(self, expr):
        """Process the sum expression."""
        self.logger.debug(f"Processing sum: {expr}")
        # Expected format: sum(n to m) expression
        pattern = r'(?i)sum\s*\(\s*(\d+)\s*to\s*(\d+)\s*\)\s*([a-zA-Z_]\w*)'
        match = re.match(pattern, expr)
        if not match:
            raise ValueError("Invalid sum format. Use 'sum(n to m) expression'.")
        start, end, sum_expr = match.groups()
        matlab_expr = f'symsum({sum_expr}, n, {start}, {end})'
        self.logger.debug(f"Converted sum to MATLAB expression: '{matlab_expr}'")
        return matlab_expr

    def _process_product(self, expr):
        """Process the product expression."""
        self.logger.debug(f"Processing product: {expr}")
        # Expected format: prod(n to m) expression
        pattern = r'(?i)prod\s*\(\s*(\d+)\s*to\s*(\d+)\s*\)\s*([a-zA-Z_]\w*)'
        match = re.match(pattern, expr)
        if not match:
            raise ValueError("Invalid product format. Use 'prod(n to m) expression'.")
        start, end, prod_expr = match.groups()
        matlab_expr = f'prod({prod_expr}, n, {start}, {end})'
        self.logger.debug(f"Converted product to MATLAB expression: '{matlab_expr}'")
        return matlab_expr

    def _add_multiplication_operators(self, expr):
        """
        Insert multiplication operators (*) where implicit multiplication is assumed.
        For example, converts '2x' to '2*x' and 'sinx' to 'sin(x)' if necessary.
        """
        self.logger.debug(f"Adding multiplication operators to expression: {expr}")
        
        # Join supported functions by '|' for regex alternation, ordered by length descendingly
        functions_pattern = '|'.join(sorted(self.SUPPORTED_FUNCTIONS, key=len, reverse=True))
        
        # Insert '*' between a number or closing parenthesis and a function or variable
        # Example: '2x' -> '2*x', '2sin(x)' -> '2*sin(x)'
        pattern = rf'(?<=[\d\)])\s*(?=({functions_pattern}|\w))'
        expr = re.sub(pattern, '*', expr, flags=re.IGNORECASE)
        
        # Insert '*' between a closing parenthesis and a function or variable
        # Example: ')x' -> ')*x', ')sin(x)' -> ')*sin(x)'
        pattern = rf'(?<=(?:{functions_pattern})\))\s*(?=\w)'
        expr = re.sub(pattern, '*', expr, flags=re.IGNORECASE)
        
        # Insert '*' between two variables or functions where appropriate
        # Example: 'x y' -> 'x*y', 'sin(x)cos(x)' -> 'sin(x)*cos(x)'
        pattern = rf'(?<=\w)\s+(?=\w)'
        expr = re.sub(pattern, '*', expr, flags=re.IGNORECASE)
        
        # Handle cases where a number is directly followed by an open parenthesis (e.g., 2(x))
        expr = re.sub(r'(\d)\s*\(', r'\1*(', expr)
        
        # Handle cases where a closing parenthesis is followed by a variable or function (e.g., )(x))
        expr = re.sub(r'\)\s*([a-zA-Z_])', r')*\1', expr)
        
        # Remove any duplicate asterisks
        expr = re.sub(r'\*+', '*', expr)
        
        self.logger.debug(f"Expression after adding multiplication operators: {expr}")
        return expr

    def _convert_trig_to_degrees(self, expr):
        """
        Convert trigonometric functions to handle degree inputs in MATLAB.
        This typically involves wrapping the input with 'deg2rad' if angle_mode is degree.
        """
        self.logger.debug(f"Converting trigonometric functions to degrees: {expr}")
        # Example: sin(30) -> sin(deg2rad(30))
        for func in self.SUPPORTED_FUNCTIONS:
            if func in ['sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'asin', 'acos', 'atan']:
                # Use regex to find function calls and wrap their arguments with deg2rad
                pattern = rf'{func}\s*\(\s*([^)]+)\s*\)'
                replacement = rf'{func}(deg2rad(\1))'
                expr = re.sub(pattern, replacement, expr, flags=re.IGNORECASE)
        self.logger.debug(f"Expression after trig conversion: {expr}")
        return expr

    def _evaluate_and_format_result(self, matlab_expression):
        self.logger.debug(f"Final MATLAB Expression: {matlab_expression}")
        
        # Create an instance of EvaluateExpression
        evaluator = EvaluateExpression(self.eng)
        result = evaluator.evaluate_matlab_expression(matlab_expression)
        
        # Convert back log to ln in the result if needed
        if hasattr(self, 'original_expr') and isinstance(result, str):
            result = result.replace('log(', 'ln(')
        
        return result

    def _display_result(self, result):
        if result is None:
            raise ValueError("Empty result returned from MATLAB")
        self.result_label.setText(f"Result: {result}")
        self.result_label.setFont(QFont("Arial", 13, QFont.Bold))
        self.logger.debug(f"Result: {result}")

    def _handle_error(self, error):
        self.logger.error(f"Error evaluating expression: {str(error)}", exc_info=True)
        QMessageBox.critical(self, "Error", f"Error evaluating expression: {str(error)}")

    def _cleanup_matlab_workspace(self):
        try:
            self.eng.eval("clear temp_result", nargout=0)
            self.logger.debug("Cleared 'temp_result' from MATLAB workspace.")
        except matlab.engine.MatlabExecutionError as me:
            self.logger.warning(f"Failed to clear 'temp_result': {me}")
        except Exception as e:
            self.logger.warning(f"Unexpected error during cleanup: {e}")