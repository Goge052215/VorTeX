import matlab.engine
import logging
import re
from matlab_interface.auto_simplify import AutoSimplify

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
        
        # Handle higher-order derivatives, e.g., 'd2/dx2 expr' -> 'diff(expr, x, 2)'
        derivative_pattern = r'd(\d*)/d([a-zA-Z])(\d*)\s+(.+)'
        def replace_derivative(match):
            order = match.group(1) or '1'
            var = match.group(2)
            expr = match.group(4)
            return f"diff({expr}, {var}, {order})"
        
        expression = re.sub(derivative_pattern, replace_derivative, expression)
        
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
        
        # Format log10 as log (only if you want to represent log10 as log)
        result_str = re.sub(r'\blog10\s*\(([^)]+)\)', r'log(\1)', result_str)
        
        # Clean up other notations
        result_str = result_str.replace('.^', '^').replace('.*', '*')
        
        self.logger.debug(f"Symbolic result after postprocessing: {result_str}")
        return result_str

    def evaluate_matlab_expression(self, matlab_expression):
        """
        Evaluate a MATLAB expression and return the result.
        
        Args:
            matlab_expression (str): The MATLAB expression to evaluate.
        
        Returns:
            str or float or list: The evaluated result.
        """
        self.logger.debug(f"Starting evaluation of MATLAB expression: '{matlab_expression}'")
        
        try:
            # Extract variables, excluding 'e'
            variables = self._extract_variables(matlab_expression)
            for var in variables:
                if var != 'e':  # Do not declare 'e' as symbolic
                    self._declare_symbolic_variable(var)
            
            # Preprocess the expression (if needed)
            processed_expr = self._preprocess_expression(matlab_expression)
            self.logger.debug(f"Preprocessed expression: '{processed_expr}'")
            
            # Execute the MATLAB expression
            matlab_cmd = f"temp_result = {processed_expr};"
            self.logger.debug(f"Executing MATLAB command: {matlab_cmd}")
            self.eng.eval(matlab_cmd, nargout=0)
            self.logger.info("Expression executed successfully in MATLAB.")

            # Check if the result is numeric
            is_numeric = self.eng.eval("isnumeric(temp_result)", nargout=1)
            self.logger.debug(f"Is the result numeric? {is_numeric}")

            if is_numeric:
                result = self.eng.eval("temp_result", nargout=1)
                self.logger.debug(f"Numeric result obtained: {result}")
                return str(float(result))

            # Check for symbolic variables
            has_symbols = self.eng.eval("~isempty(symvar(temp_result))", nargout=1)
            self.logger.debug(f"Does the result have symbolic variables? {has_symbols}")

            if has_symbols:
                result_str = self.eng.eval("char(temp_result)", nargout=1)
                # Attempt to simplify the symbolic result
                simplified_result = self.simplifier.simplify_expression(result_str)
                return simplified_result
            else:
                result = self.eng.eval("double(temp_result)", nargout=1)
                self.logger.debug(f"Double result obtained: {result}")
                return str(float(result))

        except matlab.engine.MatlabExecutionError as me:
            self.logger.error(f"MATLAB Execution Error: {me}", exc_info=True)
            raise ValueError(f"MATLAB Error: {me}") from me
        except Exception as e:
            self.logger.error(f"Unexpected Error: {e}", exc_info=True)
            raise ValueError(f"Unexpected Error: {e}") from e
        finally:
            try:
                # Clean up workspace
                self.eng.eval("clear temp_result", nargout=0)
                self.logger.debug("Cleared 'temp_result' from MATLAB workspace.")
            except Exception as e:
                self.logger.warning(f"Error during cleanup: {e}")

    def _extract_variables(self, expression):
        """
        Extract variable names from the MATLAB expression while ignoring function names.

        Args:
            expression (str): The MATLAB expression.

        Returns:
            set: A set of variable names.
        """
        # Remove derivative operators and function names
        expression_clean = re.sub(r'd\^?\d*/d[a-zA-Z]+', '', expression)
        expression_clean = re.sub(r'\b(sin|cos|tan|log|exp|sqrt|abs|sind|cosd|tand)\b', '', expression_clean)

        # Regular expression to find variable names (one or more letters)
        variables = set(re.findall(r'\b([a-zA-Z]+)\b', expression_clean))
        # Remove MATLAB reserved keywords and function names if necessary
        reserved_keywords = {'int', 'diff', 'syms', 'log', 'sin', 'cos', 'tan', 'exp', 'sqrt', 'abs', 'sind', 'cosd', 'tand'}
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

    def _postprocess_result(self, result_str, is_numeric):
        """
        Postprocess the MATLAB result.
        
        Args:
            result_str (str): The result obtained from MATLAB.
            is_numeric (bool): Indicates if the result is numeric.
        
        Returns:
            str or float: The postprocessed result.
        """
        if is_numeric:
            return str(float(result_str))
        else:
            return result_str

    def __del__(self):
        try:
            self.eng.quit()
            self.logger.info("MATLAB engine terminated")
        except:
            pass