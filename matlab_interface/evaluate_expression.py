import matlab.engine
import logging
import re
from functools import lru_cache
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
        self._compile_patterns()
        self._initialize_workspace()
        self._symbolic_vars = set()

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

    def _compile_patterns(self):
        """
        Precompile regular expressions for efficiency.
        """
        # Precompile regex patterns
        self.ln_pattern = re.compile(r'\bln\s*\(')
        # Update trig patterns to only match explicit degree functions
        self.trig_patterns = {
            re.compile(r'\bsind\s*\('): 'sind(',
            re.compile(r'\bcosd\s*\('): 'cosd(',
            re.compile(r'\btand\s*\('): 'tand(',
            re.compile(r'\barcsind\s*\('): 'asind(',
            re.compile(r'\barccosd\s*\('): 'acosd(',
            re.compile(r'\barctand\s*\('): 'atand('
        }
        self.log_e_pattern = re.compile(r'log\s*\(\s*E\s*,\s*([^,)]+)\)')
        self.log_base_pattern = re.compile(r'log\s*\(\s*(\d+)\s*,\s*([^,)]+)\)')
        self.mul_pattern = re.compile(r'(\d)([a-zA-Z])')

    def _initialize_workspace(self):
        """
        Initialize the MATLAB workspace with necessary constants and symbolic variables.
        """
        init_cmd = """
        pi = pi;      % Use MATLAB's built-in pi constant
        e = exp(1);   % Define e
        syms x;       % Define x as symbolic
        """
        self.eng.eval(init_cmd, nargout=0)
        self.logger.debug("Initialized e and pi in MATLAB workspace")

    def _preprocess_expression(self, expression):
        """
        Preprocess the expression before MATLAB evaluation.
        
        Args:
            expression (str): The input expression.
            
        Returns:
            str: The preprocessed expression.
        """
        original_expr = expression
        # Use compiled regex patterns
        expression = self.ln_pattern.sub('log(', expression)
        
        # Only convert to degree functions if explicitly specified with 'd' suffix
        # e.g., 'sind(x)' stays as 'sind(x)', but 'sin(x)' stays as 'sin(x)'
        for trig_regex, degree_func in self.trig_patterns.items():
            expression = trig_regex.sub(degree_func, expression)
        
        expression = self.log_e_pattern.sub(r'log(\1)', expression)
        expression = self.log_base_pattern.sub(r'log(\2)/log(\1)', expression)
        expression = self.mul_pattern.sub(r'\1*\2', expression)
        
        # Handle exponential expressions without adding extra parenthesis
        if 'e^' in expression:
            expression = expression.replace('e^', 'exp(')
            if not expression.endswith(')'):
                expression += ')'
        
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
        if not result_str:
            return "0"
                
        if is_numeric:
            self.logger.debug(f"Numeric result returned: {result_str}")
            try:
                float_val = float(result_str)
                return f"{float_val:.6f}".rstrip('0').rstrip('.')
            except ValueError:
                pass
        
        # Handle special cases
        special_cases = {
            'inf': '∞',
            '+inf': '∞',
            '-inf': '-∞',
            'nan': 'undefined'
        }
        result_str = special_cases.get(result_str.lower(), result_str)
        
        # Clean up symbolic results
        result_str = result_str.replace('*1.0', '').replace('1.0*', '').replace('.0', '')
        
        # Convert exp(x) to e^x and other symbolizations
        symbolizations = {
            'exp(': 'e^',
            'asin(': 'arcsin(',
            'acos(': 'arccos(',
            'atan(': 'arctan(',
            'log(': 'ln('
        }
        
        # Handle trigonometric functions with proper parentheses
        trig_funcs = ['sin', 'cos', 'tan', 'csc', 'sec', 'cot']
        for func in trig_funcs:
            # Match the function and its argument, preserving nested parentheses
            pattern = f'{func}\\(([^()]*(?:\\([^()]*\\)[^()]*)*)\\)'
            matches = list(re.finditer(pattern, result_str))
            for match in reversed(matches):  # Process from right to left
                full_match = match.group(0)
                arg = match.group(1)
                # Ensure proper parentheses around the argument
                if '(' not in arg and ')' not in arg:
                    result_str = result_str.replace(full_match, f'{func}({arg})')
        
        for old, new in symbolizations.items():
            result_str = result_str.replace(old, new)
        
        # Handle fractions and symbolic forms
        if '/' in result_str:
            result_str = re.sub(r'\s+', '', result_str)
            result_str = result_str.replace(')(', ')*(')
        
        # Remove any unmatched closing parenthesis
        if result_str.count('(') < result_str.count(')'):
            result_str = result_str.rstrip(')')
        
        self.logger.debug(f"Symbolic result after postprocessing: {result_str}")
        return result_str

    @lru_cache(maxsize=128)
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

        # Combine patterns to remove derivatives and functions in one step
        expression_clean = re.sub(r'd\^?\d*/d[a-zA-Z]+|\b(?:sin|cos|tan|log|exp|sqrt|abs|sind|cosd|tand)\b', '', expression)
        
        # Extract variable names (one or more letters)
        variables = set(re.findall(r'\b[a-zA-Z]+\b', expression_clean))
        
        # Remove MATLAB reserved keywords and function names
        reserved_keywords = {'int', 'diff', 'syms', 'log', 'sin', 'cos', 'tan', 
                             'exp', 'sqrt', 'abs', 'sind', 'cosd', 'tand', 'symsum', 
                             'prod', 'solve'}
        variables = variables - reserved_keywords
        self.logger.debug(f"Extracted variables from expression: {variables}")
        return sorted(variables)

    @lru_cache(maxsize=128)
    def _declare_symbolic_variable(self, var):
        """
        Declare a variable as symbolic in MATLAB.
        
        Args:
            var (str): Variable name.
        """
        if var in self._symbolic_vars:
            return
        declaration_cmd = f"syms {var}"
        self.logger.debug(f"Declaring symbolic variable in MATLAB: {declaration_cmd}")
        self.eng.eval(declaration_cmd, nargout=0)
        self._symbolic_vars.add(var)
        self.logger.debug(f"Declared symbolic variable: {var}")

    def evaluate_matlab_expression(self, expression):
        """Evaluate a MATLAB expression."""
        try:
            # Preprocess the expression
            preprocessed_expr = self._preprocess_expression(expression)
            
            # Define known function names to exclude from variable declarations
            function_names = {
                'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 
                'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
                'sind', 'cosd', 'tand', 'asind', 'acosd', 'atand',
                'log', 'exp', 'sqrt', 'abs', 'csc', 'sec', 'cot',
                'acsc', 'asec', 'acot', 'csch', 'sech', 'coth',
                'acsch', 'asech', 'acoth'
            }
            
            # Extract and declare symbolic variables
            variables = self._extract_variables(preprocessed_expr)
            # Filter out function names from variables
            variables = [var for var in variables if var not in function_names]
            
            for var in variables:
                self._declare_symbolic_variable(var)
            
            # Create the MATLAB command
            matlab_cmd = f"temp_result = {preprocessed_expr};"
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
                        result = f"{float_val:.6f}".rstrip('0').rstrip('.')
                except ValueError:
                    pass  # Not a number, keep original string
                
            else:
                result = str(self.eng.eval("temp_result", nargout=1))
                # Format numeric results
                try:
                    float_val = float(result)
                    if float_val != int(float_val):  # If it's truly a decimal
                        # Ensure the leading zero is preserved
                        result = f"{float_val:.6f}".rstrip('0').rstrip('.')
                except ValueError:
                    pass  # Not a number, keep original string

            self.logger.debug(f"Raw MATLAB result: {result}")

            # Post-process the result
            result = self._postprocess_result(result, is_numeric=not is_symbolic)
            self.logger.debug(f"Final result: {result}")

            return result

        except matlab.engine.EngineError as e:
            self.logger.error(f"MATLAB Engine Error: {e}")
            return "MATLAB Engine Error"
        except re.error as e:
            self.logger.error(f"Regex Error: {e}")
            return "Regex Error"
        except Exception as e:
            self.logger.error(f"Unexpected Error: {e}")
            return "Unexpected Error"

        finally:
            # Clean up the workspace in a single call
            clean_cmd = "clear temp_result temp_decimal"
            self.eng.eval(clean_cmd, nargout=0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.eng.quit()
            self.logger.info("MATLAB engine terminated")
        except Exception as e:
            self.logger.error(f"Error terminating MATLAB engine: {e}")

    def __del__(self):
        try:
            self.eng.quit()
            self.logger.info("MATLAB engine terminated")
        except:
            pass

    def _clean_expression(self, expr):
        """Clean up the expression."""
        if not expr:
            return expr
        
        self.logger.debug(f"Original expression before cleaning: '{expr}'")
        
        # Convert 'exp(x)' to 'e^x' for display, ensuring no extra parenthesis
        expr = re.sub(r'exp\((.*?)\)', r'e^\1', expr)
        
        # Other cleaning operations...
        expr = expr.replace('*1.0', '').replace('1.0*', '')
        expr = expr.replace('log(', 'ln(')
        expr = re.sub(r'\b1\*\s*([a-zA-Z])', r'\1', expr)
        
        self.logger.debug(f"Final cleaned expression: '{expr}'")
        
        return expr