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
            'log', 'exp', 'sqrt', 'factorial', 'solve'
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
        
        # Handle higher-order derivatives, e.g., 'd2/dx2 expr' -> 'diff(expr, x, 2)'
        derivative_pattern = r'd(\d*)/d([a-zA-Z])(\d*)\s+(.+)'
        def replace_derivative(match):
            order = match.group(1) or '1'
            var = match.group(2)
            expr = match.group(4)
            # Ensure multiplication is explicit within the derivative expression
            expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)
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

    def evaluate_matlab_expression(self, expression):
        """
        Evaluate a MATLAB expression and return the result.
        
        Args:
            expression (str): The MATLAB expression to evaluate.
        
        Returns:
            str or float or list: The evaluated result.
        """
        self.logger.debug(f"Starting evaluation of MATLAB expression: '{expression}'")
        
        try:
            # Convert combinatorial expressions before evaluation
            expression = ExpressionShortcuts.convert_combinatorial_expression(expression)
            self.logger.debug(f"After shortcut conversion: {expression}")

            expression = re.sub(r'(\d+)C(\d+)', r'nchoosek(\1, \2)', expression)  # Handle nCr pattern
            expression = re.sub(r'binom\s*\((\d+)\s*,\s*(\d+)\)', r'nchoosek(\1, \2)', expression)
            expression = re.sub(r'nCr\s*\((\d+)\s*,\s*(\d+)\)', r'nchoosek(\1, \2)', expression)
            expression = re.sub(r'nPr\s*\((\d+)\s*,\s*(\d+)\)', r'nchoosek(\1, \2)', expression)

            self.logger.debug(f"After combinatorial conversion: {expression}")

            # Convert logarithms before evaluation
            expression = ExpressionShortcuts._convert_logarithms(expression)
            self.logger.debug(f"After logarithm conversion: {expression}")

            # Special handling for solve function
            if 'solve' in expression:
                # Always symbolize x for equations
                self.eng.eval("syms x", nargout=0)
                self.logger.debug("Symbolized x for equation solving")
                
                # Evaluate solve expression
                matlab_cmd = f"temp_result = {expression};"
                self.logger.debug(f"Executing solve command: {matlab_cmd}")
                self.eng.eval(matlab_cmd, nargout=0)
                
                # Convert solutions to double array
                self.eng.eval("temp_double = double(temp_result);", nargout=0)
                solutions = self.eng.eval("temp_double", nargout=1)
                
                # Format solutions
                if isinstance(solutions, float):
                    return f"x = {solutions}"
                else:
                    formatted_solutions = []
                    for i, sol in enumerate(solutions, 1):
                        formatted_solutions.append(f"x{i} = {sol}")
                    return '\n'.join(formatted_solutions)

            # Special handling for nchoosek
            if 'nchoosek' in expression:
                self.logger.debug("Evaluating combinatorial expression numerically")
                result = self.eng.eval(expression, nargout=1)
                return str(float(result))

            # Special handling for logarithms
            if any(log in expression for log in ['log', 'log10']):
                # Ensure x is symbolic for logarithm evaluation
                self.eng.eval("syms x", nargout=0)
                self.logger.debug("Symbolized x for logarithm evaluation")

            # Extract variables (excluding built-in functions)
            variables = self._extract_variables(expression)
            variables = variables - self.matlab_functions
            
            # Declare variables as symbolic in MATLAB
            for var in variables:
                self.logger.debug(f"Declaring symbolic variable in MATLAB: syms {var}")
                self.eng.eval(f"syms {var}", nargout=0)
                self.logger.debug(f"Declared symbolic variable: {var}")
            
            processed_expr = self._preprocess_expression(expression)
            self.logger.debug(f"Preprocessed expression: '{processed_expr}'")
            
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

        except Exception as e:
            self.logger.error(f"MATLAB Execution Error: {str(e)}")
            raise
            
        finally:
            # Clean up temporary variable
            self.eng.eval("clear temp_result temp_double", nargout=0)

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