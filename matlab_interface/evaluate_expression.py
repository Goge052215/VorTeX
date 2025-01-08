import matlab.engine
import logging
import re
from matlab_interface.auto_simplify import AutoSimplify
from latex_pack.shortcut import ExpressionShortcuts

class EvaluateExpression:
    def __init__(self, eng):
        """
        Initialize the EvaluateExpression class with a MATLAB engine instance.
        
        Args:
            eng (matlab.engine.MatlabEngine): An active MATLAB engine session.
        """
        self.eng = eng
        self.logger = logging.getLogger(__name__)
        self._configure_logger()
        self.simplifier = AutoSimplify(eng)
        
        # Define built-in MATLAB functions that shouldn't be treated as variables
        self.matlab_functions = {
            'nchoosek', 'diff', 'int', 'sin', 'cos', 'tan', 
            'log', 'exp', 'sqrt', 'factorial', 'solve', 'symsum', 'prod'
        }

    def _configure_logger(self):
        """
        Configure the logger for the EvaluateExpression class.
        """
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)

    def _preprocess_expression(self, expression):
        """
        Preprocess the expression before MATLAB evaluation.
        
        Args:
            expression (str): The input expression.
            
        Returns:
            str: The preprocessed expression.
        """
        original_expr = expression
        
        # Convert ln(x) to log(x) for MATLAB processing
        expression = re.sub(r'\bln\s*\(', 'log(', expression)
        
        # Convert trigonometric functions to degree mode
        trig_functions = {
            r'\bsin\s*\(': 'sind(',
            r'\bcos\s*\(': 'cosd(',
            r'\btan\s*\(': 'tand(',
            r'\barcsin\s*\(': 'asind(',
            r'\barccos\s*\(': 'acosd(',
            r'\barctan\s*\(': 'atand('
        }
        
        for trig_func, degree_func in trig_functions.items():
            expression = re.sub(trig_func, degree_func, expression)
        
        # Convert log(E, x) to log(x) for natural logarithm
        expression = re.sub(r'log\s*\(\s*E\s*,\s*([^,)]+)\)', r'log(\1)', expression)
        
        # Handle log with different bases, e.g., log(b, x) -> log(x)/log(b)
        expression = re.sub(r'log\s*\(\s*(\d+)\s*,\s*([^,)]+)\)', r'log(\2)/log(\1)', expression)
        
        # Ensure multiplication is explicit, e.g., '4x' becomes '4*x'
        expression = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expression)

        # Handle sum and prod functions
        sum_pattern = r'\bsum\s*\(([^)]+)\)'
        prod_pattern = r'\bprod\s*\(([^)]+)\)'
        expression = re.sub(sum_pattern, r'sum(\1)', expression)
        expression = re.sub(prod_pattern, r'prod(\1)', expression)

        # Handle LaTeX-style sum expressions: \sum_{a}^{b} f(x)
        latex_sum_pattern = r'\\sum_{([^}]+)}\^{([^}]+)}\s*([^$]+)'
        def replace_latex_sum(match):
            lower_limit = match.group(1).strip()
            upper_limit = match.group(2).strip()
            function_expr = match.group(3).strip()
            # Ensure multiplication is explicit in the function expression
            function_expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', function_expr)
            return f"symsum({function_expr}, x, {lower_limit}, {upper_limit})"
        
        expression = re.sub(latex_sum_pattern, replace_latex_sum, expression)

        # Handle LaTeX-style product expressions: \prod_{a}^{b} f(x)
        latex_prod_pattern = r'\\prod_{([^}]+)}\^{([^}]+)}\s*([^$]+)'
        def replace_latex_prod(match):
            lower_limit = match.group(1).strip()
            upper_limit = match.group(2).strip()
            function_expr = match.group(3).strip()
            # Ensure multiplication is explicit in the function expression
            function_expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', function_expr)
            return f"prod({function_expr}, x, {lower_limit}, {upper_limit})"
        
        expression = re.sub(latex_prod_pattern, replace_latex_prod, expression)

        # Regular expression to match 'int expression dvariable'
        integral_pattern = r'int\s+(.+)\s+d([a-zA-Z])'
        def replace_integral(match):
            expr = match.group(1).strip()
            var = match.group(2).strip()
            # Replace '^' with '.^' for MATLAB compatibility
            expr = expr.replace('^', '.^')
            return f'int({expr}, {var})'
        
        # Handle higher-order derivatives, e.g., 'd2/dx2 expr' -> 'diff(expr, x, 2)'
        derivative_pattern = r'd/d([a-zA-Z])\s*([^$]+)'
        def replace_derivative(match):
            var = match.group(1)
            expr = match.group(2).strip()
            # Ensure multiplication is explicit in the expression
            expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)
            # Convert Python-style power operator to MATLAB
            expr = expr.replace('^', '.^')
            return f"diff({expr}, {var})"
        
        expression = re.sub(integral_pattern, derivative_pattern, replace_derivative, replace_integral, expression)
        
        self.logger.debug(f"Converted '{original_expr}' to '{expression}'")
        return expression

    def _postprocess_result(self, result, is_numeric=True):
        """
        Clean up the result for display.
        
        Args:
            result: The result to process
            is_numeric: Boolean indicating if the result is numeric
        """
        if not result:
            return "0"
            
        if isinstance(result, (int, float)):
            return f"{float(result):.6f}".rstrip('0').rstrip('.')
            
        result = str(result)
        
        # Handle special cases
        if result.lower() in ['inf', '+inf']:
            return '∞'
        elif result.lower() == '-inf':
            return '-∞'
        elif result.lower() == 'nan':
            return 'undefined'
            
        # Try to convert to float for numeric results
        try:
            if is_numeric and result.replace('.', '').replace('-', '').isdigit():
                return f"{float(result):.6f}".rstrip('0').rstrip('.')
        except:
            pass
        
        # Clean up symbolic results
        result = result.replace('*1.0', '')
        result = result.replace('1.0*', '')
        result = result.replace('.0', '')
        
        # Convert exp(x) to e^x
        result = re.sub(r'exp\((.*?)\)', r'e^\1', result)
        
        # Symbolization steps
        result = result.replace('asin(', 'arcsin(')
        result = result.replace('acos(', 'arccos(')
        result = result.replace('atan(', 'arctan(')
        result = result.replace('log(', 'ln(')
        
        # Handle fractions and other symbolic forms
        if '/' in result:
            result = result.replace(' ', '')
            result = result.replace(')(', ')*(')
        
        return result

    def evaluate_matlab_expression(self, expression):
        """
        Evaluate a MATLAB expression and return the result.
        """
        try:
            # Initialize constants
            init_cmd = """
            pi = pi;      % Use MATLAB's built-in pi constant
            e = exp(1);   % Define e
            syms x;       % Define x as symbolic
            """
            self.eng.eval(init_cmd, nargout=0)
            self.logger.debug("Initialized e and pi in MATLAB workspace")

            # Special handling for constants and simple expressions
            if expression.strip() == 'e':
                return 'e'
            elif expression.strip() == 'pi':
                return 'pi'
            
            # Create the MATLAB command
            matlab_cmd = f"temp_result = {expression};"
            self.logger.debug(f"Executing MATLAB command: {matlab_cmd}")
            self.eng.eval(matlab_cmd, nargout=0)

            # Check if the result is symbolic
            is_symbolic = self.eng.eval("isa(temp_result, 'sym')", nargout=1)
            self.logger.debug(f"Is the result symbolic? {is_symbolic}")

            if is_symbolic:
                # Get the result as a string first
                result = self.eng.eval("char(temp_result)", nargout=1)
                
                # Check if it's a complicated fraction
                if '/' in result:
                    numerator, denominator = result.split('/')
                    # If either number is too long (more than 10 digits), convert to decimal
                    if len(numerator) > 10 or len(denominator) > 10:
                        # Convert to decimal with 6 decimal places
                        self.eng.eval("temp_decimal = double(temp_result);", nargout=0)
                        decimal_result = float(self.eng.eval("temp_decimal", nargout=1))
                        result = f"{decimal_result:.6f}"
                
                # Try to convert to float and format if it's a decimal number
                try:
                    float_val = float(result)
                    if float_val != int(float_val):  # If it's truly a decimal
                        # Ensure the leading zero is preserved
                        result = f"{float_val:.6f}"
                except ValueError:
                    pass  # Not a number, keep original string
                
            else:
                result = str(self.eng.eval("temp_result", nargout=1))
                # Format numeric results
                try:
                    float_val = float(result)
                    if float_val != int(float_val):  # If it's truly a decimal
                        # Ensure the leading zero is preserved
                        result = f"{float_val:.6f}"
                except ValueError:
                    pass  # Not a number, keep original string

            self.logger.debug(f"Raw MATLAB result: {result}")

            # Post-process the result
            result = self._postprocess_result(result, is_numeric=not is_symbolic)
            self.logger.debug(f"Final result: {result}")

            return result

        except Exception as e:
            self.logger.error(f"Error evaluating expression: {e}")
            return str(e)

        finally:
            # Clean up the workspace
            self.eng.eval("clear temp_result temp_decimal", nargout=0)

    def _extract_variables(self, expression):
        """
        Extract variables from the expression for MATLAB evaluation.
        
        Args:
            expression (str): The input expression.
        
        Returns:
            list: A list of variable names.
        """
        if not isinstance(expression, str):
            self.logger.error("Expression is not a string")
            raise TypeError("Expression must be a string")

        # Clean the expression to remove derivative notation
        expression_clean = re.sub(r'd\^?\d*/d[a-zA-Z]+', '', expression)
        expression_clean = re.sub(r'\b(sin|cos|tan|log|exp|sqrt|abs|sind|cosd|tand)\b', '', expression_clean)

        # Regular expression to find variable names (one or more letters)
        variables = set(re.findall(r'\b([a-zA-Z]+)\b', expression_clean))
        # Remove MATLAB reserved keywords and function names if necessary
        reserved_keywords = {'int', 'diff', 'syms', 'log', 'sin', 'cos', 'tan', 
                             'exp', 'sqrt', 'abs', 'sind', 'cosd', 'tand', 'symsum', 'prod'}
        variables = variables - reserved_keywords
        self.logger.debug(f"Extracted variables from expression: {variables}")
        return variables

    def _declare_symbolic_variable(self, var):
        """
        Declare a variable as symbolic in MATLAB.
        
        Args:
            var (str): Variable name.
        """
        declaration_cmd = f"syms {var}"
        self.logger.debug(f"Declaring symbolic variable in MATLAB: {declaration_cmd}")
        self.eng.eval(declaration_cmd, nargout=0)
        self.logger.debug(f"Declared symbolic variable: {var}")

    def _preprocess_expression(self, expression):
        """
        Preprocess the expression before sending to MATLAB.
        
        Args:
            expression (str): The MATLAB expression.
        
        Returns:
            str: Preprocessed expression.
        """
        original_expr = expression
        # Convert ln(x) to log(x) for MATLAB processing
        expression = re.sub(r'\bln\s*\(', 'log(', expression)
        
        # Convert trigonometric functions to degree mode
        trig_functions = {
            r'\bsin\s*\(': 'sind(',
            r'\bcos\s*\(': 'cosd(',
            r'\btan\s*\(': 'tand(',
            r'\barcsin\s*\(': 'asind(',
            r'\barccos\s*\(': 'acosd(',
            r'\barctan\s*\(': 'atand('
        }
        
        for trig_func, degree_func in trig_functions.items():
            expression = re.sub(trig_func, degree_func, expression)
        
        # Convert log(E, x) to log(x) for natural logarithm
        expression = re.sub(r'log\s*\(\s*E\s*,\s*([^,)]+)\)', r'log(\1)', expression)
        
        # Handle log with different bases, e.g., log(b, x) -> log(x)/log(b)
        expression = re.sub(r'log\s*\(\s*(\d+)\s*,\s*([^,)]+)\)', r'log(\2)/log(\1)', expression)
        
        # Ensure multiplication is explicit, e.g., '4x' becomes '4*x'
        expression = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expression)
        
        self.logger.debug(f"Converted '{original_expr}' to '{expression}'")
        return expression

    def _postprocess_result(self, result_str, is_numeric=False):
        """
        Format the result string for display.
        
        Args:
            result_str (str): The result string from MATLAB.
            is_numeric (bool): Whether the result is numeric.
            
        Returns:
            str: The formatted result string.
        """
        if is_numeric:
            self.logger.debug(f"Numeric result returned: {result_str}")
            return result_str
            
        # Convert natural log back to ln
        result_str = re.sub(r'\blog\s*\(([^,)]+)\)', r'ln(\1)', result_str)
        
        # Clean up other notations
        result_str = result_str.replace('.^', '^').replace('.*', '*')
        
        self.logger.debug(f"Symbolic result after postprocessing: {result_str}")
        return result_str

    def __del__(self):
        try:
            self.eng.quit()
            self.logger.info("MATLAB engine terminated")
        except:
            pass