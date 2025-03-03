import logging
import re
from typing import Optional, Union

class AutoSimplify:
    """
    A class to handle automatic simplification of mathematical expressions using MATLAB.
    """
    
    def __init__(self, eng):
        """
        Initialize the AutoSimplify class with a MATLAB engine instance.
        
        Args:
            eng (matlab.engine.MatlabEngine): An active MATLAB engine session.
        """
        self.eng = eng
        self.logger = logging.getLogger(__name__)
        self._configure_logger()
        
        # Initialize MATLAB symbolic variables
        self._init_matlab_symbolic_vars()
        
        # Define mathematical constants and their decimal approximations
        self.MATH_CONSTANTS = {
            # Euler's number and powers
            'e': 2.71828182845904523536,
            'e^2': 7.38905609893065,
            'e^-1': 0.36787944117144233,
            '1/e': 0.36787944117144233,
            'e^(1/2)': 1.6487212707001282,
            'sqrt(e)': 1.6487212707001282,
            'e^(1/3)': 1.3956124250860895,
            
            # Pi and related values
            'π': 3.14159265358979323846,
            'π/2': 1.5707963267948966,
            'π/3': 1.0471975511965976,
            'π/4': 0.7853981633974483,
            'π/6': 0.5235987755982988,
            '2π': 6.283185307179586,
            '3π': 9.42477796076938,
            '4π': 12.566370614359172,
            'π^2': 9.869604401089358,
            'π^2/6': 1.6449340668482264,
            '√π': 1.7724538509055159,
            'sqrt(π)': 1.7724538509055159,
            
            # Log values
            'ln(2)': 0.6931471805599453,
            'ln(10)': 2.302585092994046,
            'log(2)': 0.6931471805599453,
            
            # Common fractions
            '1/2': 0.5,
            '1/3': 0.3333333333333333,
            '2/3': 0.6666666666666666,
            '1/4': 0.25,
            '3/4': 0.75,
            '1/6': 0.16666666666666666,
            '5/6': 0.8333333333333334,
            
            # Square roots
            'sqrt(2)': 1.4142135623730951,
            'sqrt(3)': 1.7320508075688772,
            'sqrt(5)': 2.2360679774997898,
            '1/sqrt(2)': 0.7071067811865475,
            'sqrt(2)/2': 0.7071067811865475,
            
            # Common trig values
            'sin(π/6)': 0.5,
            'sin(π/4)': 0.7071067811865475,
            'sin(π/3)': 0.8660254037844386,
            'cos(π/6)': 0.8660254037844386,
            'cos(π/4)': 0.7071067811865475,
            'cos(π/3)': 0.5,
            'tan(π/4)': 1.0,
            'tan(π/6)': 0.5773502691896257,
            'tan(π/3)': 1.7320508075688767
        }
        
        # Define special calculus patterns
        self.CALCULUS_PATTERNS = {
            # Integration patterns
            'e^2*x^2/2': {
                'pattern': r'(?:3\.694|3\.695|3\.69)\d*\*x\^2',
                'exact': '(e^2)*x^2/2'
            },
            'e*x^2/2': {
                'pattern': r'(?:1\.359|1\.36)\d*\*x\^2',
                'exact': 'e*x^2/2'
            },
            'x^2/2': {
                'pattern': r'(?:0\.5|0\.50)\d*\*x\^2',
                'exact': 'x^2/2'
            },
            'x^3/3': {
                'pattern': r'(?:0\.333|0\.33)\d*\*x\^3',
                'exact': 'x^3/3'
            },
            'x^4/4': {
                'pattern': r'(?:0\.25|0\.250)\d*\*x\^4',
                'exact': 'x^4/4'
            },
            'x^n/n': {
                'pattern': r'(\d+\.\d+)\*x\^(\d+)',
                'func': lambda match: self._check_power_pattern(match)
            },
            
            # π patterns
            'π*x': {
                'pattern': r'(?:3\.14|3\.141|3\.1415|3\.14159)\d*\*x',
                'exact': 'π*x'
            },
            'π/2*x': {
                'pattern': r'(?:1\.57|1\.570|1\.5707|1\.5708)\d*\*x',
                'exact': 'π/2*x'
            },
            '2π*x': {
                'pattern': r'(?:6\.28|6\.283|6\.2831|6\.28318)\d*\*x',
                'exact': '2π*x'
            },
            
            # Trig function patterns
            'sin': {
                'pattern': r'(?:0\.5|0\.50|0\.500)\*sin\(([^)]+)\)',
                'exact': '(1/2)*sin(\\1)'
            },
            'cos': {
                'pattern': r'(?:0\.5|0\.50|0\.500)\*cos\(([^)]+)\)',
                'exact': '(1/2)*cos(\\1)'
            },
            'neg_sin': {
                'pattern': r'\-(?:0\.5|0\.50|0\.500)\*sin\(([^)]+)\)',
                'exact': '-(1/2)*sin(\\1)'
            },
            'neg_cos': {
                'pattern': r'\-(?:0\.5|0\.50|0\.500)\*cos\(([^)]+)\)',
                'exact': '-(1/2)*cos(\\1)'
            }
        }

    def _configure_logger(self):
        """Configure the logger for the AutoSimplify class."""
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)  # Set to INFO level for general use
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _init_matlab_symbolic_vars(self):
        """Initialize common symbolic variables in MATLAB workspace."""
        try:
            # Initialize common variables
            self.eng.eval("syms x y z t a b c n real", nargout=0)
            
            self.eng.eval("e = exp(1);", nargout=0)
            
            self.logger.debug("Initialized symbolic variables in MATLAB workspace")
        except Exception as e:
            self.logger.error(f"Error initializing symbolic variables: {e}")
            raise

    def simplify_expression(self, expr):
        """Simplify the given mathematical expression."""
        try:
            # Convert expression to string
            expr_str = str(expr)
            
            # If the input is already a multi-line result, just round it
            if '\n' in expr_str and ('=' in expr_str):
                return self._round_multiline_result(expr_str)
            
            # Ensure proper MATLAB syntax
            expr_str = expr_str.replace('=', '==')
            
            # Check if the expression contains 'Inf'
            if 'Inf' in expr_str or 'inf' in expr_str:
                return 'Inf'
            
            # Check for special trigonometric values
            trig_pattern = r'(sin|cos|tan)\s*\(\s*(pi|pi/\d+|\d+\s*\*\s*pi|\d+\s*\*\s*pi/\d+)\s*\)'
            if re.search(trig_pattern, expr_str):
                self.logger.debug(f"Detected trigonometric expression with special value: {expr_str}")
                # Try to evaluate directly first
                try:
                    self.eng.eval(f"temp_result = double({expr_str});", nargout=0)
                    result_value = float(self.eng.eval("temp_result", nargout=1))
                    self.eng.eval("clear temp_result", nargout=0)
                    
                    # Check if the result is close to a known value
                    return self.round_numeric_value(result_value)
                except Exception as e:
                    self.logger.debug(f"Error in direct evaluation of trig expression: {e}")
                    # Continue with symbolic simplification
            
            # Check for exponential expressions
            if 'exp(' in expr_str:
                expr_str = re.sub(r'exp\((.*?)\)', r'e^\1', expr_str)
                
                # Try to simplify the exponential expression
                try:
                    self.eng.eval(f"syms x; temp_result = simplify({expr_str});", nargout=0)
                    result = self.eng.eval("char(temp_result)", nargout=1)
                    self.eng.eval("clear temp_result", nargout=0)
                    
                    if result:
                        return result
                except Exception as e:
                    self.logger.debug(f"Error simplifying exponential expression: {e}")
                
                return expr_str
            
            # Try different simplification approaches
            try:
                # First try with simplify
                self.eng.eval(f"syms x; temp_result = simplify({expr_str});", nargout=0)
                result1 = self.eng.eval("char(temp_result)", nargout=1)
                
                # Then try with vpa(simplify())
                self.eng.eval("temp_result = vpa(simplify(temp_result), 4);", nargout=0)
                result2 = self.eng.eval("char(temp_result)", nargout=1)
                
                # Choose the shorter/simpler result
                result = result1 if len(result1) <= len(result2) else result2
                
                # If the result is still complex, try factor
                if '+' in result and '-' in result and '*' in result:
                    self.eng.eval("temp_result = factor(temp_result);", nargout=0)
                    factored_result = self.eng.eval("char(temp_result)", nargout=1)
                    if len(factored_result) < len(result):
                        result = factored_result
                
                self.eng.eval("clear temp_result", nargout=0)
            except Exception as e:
                self.logger.error(f"Error during symbolic simplification: {e}")
                # Try direct numerical evaluation as fallback
                try:
                    self.eng.eval(f"temp_result = double({expr_str});", nargout=0)
                    result_value = float(self.eng.eval("temp_result", nargout=1))
                    self.eng.eval("clear temp_result", nargout=0)
                    return self.round_numeric_value(result_value)
                except:
                    return expr_str
            
            if 'e^' in result or 'exp(' in result:
                result = result.replace('exp(', 'e^')
                result = result.rstrip(')')
            
            # Round the result
            try:
                # Try to convert to float for rounding
                float_val = float(result)
                rounded_result = self.round_numeric_value(float_val)
            except ValueError:
                # Not a simple float, use regex to find numbers or keep as is
                if re.search(r'\d+\.\d+', result):
                    rounded_result = re.sub(
                        r'(\d+\.\d+)',
                        lambda match: self.round_numeric_value(float(match.group(0))),
                        result
                    )
                else:
                    rounded_result = result
            
            # Clean up the result
            rounded_result = self._clean_expression(rounded_result)
            
            return rounded_result
            
        except Exception as e:
            self.logger.error(f"Error during simplification: {e}")
            return expr

    def _round_multiline_result(self, result):
        """Round numbers in a multi-line result using intelligent rounding."""
        try:
            lines = result.split('\n')
            rounded_lines = []
            
            for line in lines:
                if '=' in line:
                    var, value = line.split('=', 1)
                    value = value.strip()
                    
                    # Handle complex numbers
                    if 'i' in value:
                        # Extract real and imaginary parts using regex
                        import re
                        # Match pattern: real_part [+-] imag_part i
                        match = re.match(r'([-\d.]+)\s*([+-])\s*([\d.]+)i', value)
                        if match:
                            real_part = float(match.group(1))
                            sign = match.group(2)
                            imag_part = float(match.group(3))
                            
                            if sign == '-':
                                imag_part = -imag_part
                                
                            # Use round_numeric_value for each part
                            rounded_real = self.round_numeric_value(real_part)
                            rounded_imag = self.round_numeric_value(abs(imag_part))
                            
                            if float(real_part) == 0:
                                if imag_part > 0:
                                    rounded_value = f"{rounded_imag}i"
                                else:
                                    rounded_value = f"-{rounded_imag}i"
                            else:
                                if imag_part > 0:
                                    rounded_value = f"{rounded_real} + {rounded_imag}i"
                                else:
                                    rounded_value = f"{rounded_real} - {rounded_imag}i"
                        else:
                            # Handle pure imaginary numbers
                            try:
                                imag_part = float(value.replace('i', ''))
                                rounded_imag = self.round_numeric_value(abs(imag_part))
                                if imag_part > 0:
                                    rounded_value = f"{rounded_imag}i"
                                else:
                                    rounded_value = f"-{rounded_imag}i"
                            except ValueError:
                                rounded_value = value  # Keep original if parsing fails
                    else:
                        # Handle real numbers
                        try:
                            num = float(value)
                            rounded_value = self.round_numeric_value(num)
                        except ValueError:
                            rounded_value = value
                    
                    rounded_lines.append(f"{var.strip()} = {rounded_value}")
                else:
                    rounded_lines.append(line)
            
            return '\n'.join(rounded_lines)
            
        except Exception as e:
            self.logger.error(f"Error rounding multiline result: {e}")
            return result

    def _round_result(self, result):
        """Round numerical values in the result to provide more meaningful representations."""
        try:
            # Handle complex numbers
            if 'i' in result:
                return self._round_multiline_result(result)
            
            # Try to convert to float and use round_numeric_value
            try:
                value = float(result)
                return self.round_numeric_value(value)
            except ValueError:
                # Not a simple float, use regex to find numbers
                rounded_result = re.sub(
                    r'(\d+\.\d+)',
                    lambda match: self.round_numeric_value(float(match.group(0))),
                    result
                )
                return rounded_result
        except Exception as e:
            self.logger.error(f"Error rounding result: {e}")
            return result

    def _clean_expression(self, expr):
        """Clean up the expression."""
        if not expr:
            return expr
            
        self.logger.debug(f"Original expression before cleaning: '{expr}'")
        
        # Common numerical values to replace with pi
        pi_replacements = {
            '3.14159265359': 'π',
            '0.0174533': 'π/180',
            '0.785398': 'π/4',
            '1.5708': 'π/2',
            '3.14159': 'π',
            '6.28319': '2π',
            'pi': 'π'
        }
        
        # Replace numerical values with pi symbols
        for num, pi_expr in pi_replacements.items():
            expr = expr.replace(num, pi_expr)
        
        # Clean up common patterns
        expr = expr.replace('*1.0', '')
        expr = expr.replace('1.0*', '')
        expr = expr.replace('*1)', ')')
        expr = expr.replace('(1*', '(')
        
        # Convert log to ln for display
        expr = expr.replace('log(', 'ln(')
        
        # Remove unnecessary multiplications by 1
        expr = re.sub(r'\b1\*\s*([a-zA-Z])', r'\1', expr)
        
        # Remove unnecessary parentheses around simple terms
        expr = re.sub(r'\(([a-zA-Z0-9]+)\)', r'\1', expr)
        
        # Format sqrt expressions
        expr = re.sub(r'(\d+)\^(1/2)', r'sqrt(\1)', expr)
        
        self.logger.debug(f"Final cleaned expression: '{expr}'")
        
        return expr
        
    @staticmethod
    def round_numeric_value(value, tolerance=1e-10):
        """
        Apply intelligent rounding to numeric values:
        1. Round to integers when very close
        2. Convert to fractions when close to common fractions
        3. Format floats appropriately otherwise
        
        Args:
            value (float): The numeric value to round
            tolerance (float): The tolerance for rounding (default: 1e-10)
            
        Returns:
            str: Formatted string representation of the value
        """
        # Handle special values
        if abs(value) < tolerance:
            return "0"
            
        # Round to integer if very close
        if abs(value - round(value)) < tolerance:
            return str(int(round(value)))
            
        # Check for common fractions
        common_fractions = [
            # Halves
            (0.5, "1/2"),
            # Quarters
            (0.25, "1/4"), (0.75, "3/4"),
            # Thirds
            (1/3, "1/3"), (2/3, "2/3"),
            # Fifths
            (0.2, "1/5"), (0.4, "2/5"), (0.6, "3/5"), (0.8, "4/5"),
            # Sixths
            (1/6, "1/6"), (5/6, "5/6"),
            # Eighths
            (1/8, "1/8"), (3/8, "3/8"), (5/8, "5/8"), (7/8, "7/8"),
            # Common square roots
            (2**0.5, "sqrt(2)"), (3**0.5, "sqrt(3)"), 
            (0.5*2**0.5, "sqrt(2)/2"), (0.5*3**0.5, "sqrt(3)/2"),
            (0.25*2**0.5, "sqrt(2)/4"), (0.25*3**0.5, "sqrt(3)/4"),
            # Pi-related values
            (3.14159265358979, "π"), (3.14159265358979/2, "π/2"), 
            (3.14159265358979/3, "π/3"), (3.14159265358979/4, "π/4"),
            (3.14159265358979/6, "π/6")
        ]
        
        for fraction_value, fraction_str in common_fractions:
            if abs(value - fraction_value) < tolerance:
                return fraction_str
                
        # Handle negative common fractions
        for fraction_value, fraction_str in common_fractions:
            if abs(value + fraction_value) < tolerance:
                return "-" + fraction_str
                
        # Handle integer + common fraction
        for fraction_value, fraction_str in common_fractions:
            integer_part = int(value)
            decimal_part = value - integer_part
            if abs(decimal_part - fraction_value) < tolerance and decimal_part > 0:
                return f"{integer_part} + {fraction_str}"
            if abs(decimal_part + fraction_value) < tolerance and decimal_part < 0:
                return f"{integer_part} - {fraction_str}"
        
        # Use _smart_round for other values (on instance method)
        # Since this is a static method, we'll handle it differently in the implementation
        
        # For other values, return formatted float
        abs_value = abs(value)
        
        if abs_value < 0.001:
            # Scientific notation for very small numbers
            return f"{value:.2e}"
        elif abs_value < 0.1:
            # More precision for small numbers
            rounded = round(value, 6)
        elif abs_value < 1:
            rounded = round(value, 5)
        elif abs_value < 10:
            rounded = round(value, 4)
        elif abs_value < 100:
            rounded = round(value, 3)
        else:
            rounded = round(value, 1)
            
        # Format with appropriate precision avoiding trailing zeros
        result = f"{rounded}".rstrip('0').rstrip('.')
        return result

    def simplify_equation(self, equation: str) -> str:
        """
        Simplify an equation (expression with equals sign).
        
        Args:
            equation (str): The equation to simplify.
            
        Returns:
            str: Simplified equation.
        """
        try:
            if '=' not in equation:
                raise ValueError("Input must be an equation containing '='")
                
            lhs, rhs = equation.split('=')
            
            # Simplify both sides
            simplified_lhs = self.simplify_expression(lhs.strip())
            simplified_rhs = self.simplify_expression(rhs.strip())
            
            return f"{simplified_lhs} = {simplified_rhs}"
            
        except Exception as e:
            self.logger.error(f"Error simplifying equation: {e}")
            raise

    def collect_terms(self, expression: str, variable: str) -> str:
        """
        Collect terms with respect to a specific variable.
        
        Args:
            expression (str): The expression to collect terms from.
            variable (str): The variable to collect terms for.
            
        Returns:
            str: Expression with collected terms.
        """
        try:
            self.logger.debug(f"Collecting terms for {variable} in: {expression}")
            
            # Process expression in MATLAB
            self.eng.eval(f"expr = {expression};", nargout=0)
            self.eng.eval(f"collected = collect(expr, {variable});", nargout=0)
            
            # Get result
            result = self.eng.eval("char(collected)", nargout=1)
            
            return self._postprocess_result(result)
            
        except Exception as e:
            self.logger.error(f"Error collecting terms: {e}")
            raise
        finally:
            self.eng.eval("clear expr collected", nargout=0)

    def factor_expression(self, expression: str) -> str:
        """
        Factor an expression.
        
        Args:
            expression (str): The expression to factor.
            
        Returns:
            str: Factored expression.
        """
        try:
            self.logger.debug(f"Factoring expression: {expression}")
            
            # Process in MATLAB
            self.eng.eval(f"expr = {expression};", nargout=0)
            self.eng.eval("factored = factor(expr);", nargout=0)
            
            # Get result
            result = self.eng.eval("char(factored)", nargout=1)
            
            return self._postprocess_result(result)
            
        except Exception as e:
            self.logger.error(f"Error factoring expression: {e}")
            raise
        finally:
            self.eng.eval("clear expr factored", nargout=0)

    def _postprocess_result(self, result):
        """
        Apply additional post-processing to the result.
        
        Args:
            result (str): The result to post-process.
            
        Returns:
            str: The post-processed result.
        """
        if not result:
            return result
            
        # Convert exponential expressions
        if 'exp(' in result:
            result = re.sub(r'exp\((.*?)\)', r'e^(\1)', result)
        
        # Format fractions
        if '/' in result:
            # Ensure proper spacing around operators
            result = re.sub(r'(\d+)/(\d+)', r'\1/\2', result)
        
        # Format powers
        result = re.sub(r'\^(\d+)', r'^{\1}', result)
        
        # Clean up unnecessary parentheses
        result = re.sub(r'\(\(([^()]+)\)\)', r'(\1)', result)
        
        return result

    def simplify_large_fractions(self, result_str):
        """
        Simplify expressions containing large fractions to improve readability.
        Uses pattern recognition to convert approximate decimal values to exact mathematical forms.
        
        Args:
            result_str (str): The result string potentially containing fractions or decimal approximations
            
        Returns:
            str: The result with simplified fractions and exact mathematical forms where possible
        """
        # If result is empty or None, return as is
        if not result_str or not isinstance(result_str, str):
            return result_str
        
        # First, check known calculus patterns using regex
        for name, info in self.CALCULUS_PATTERNS.items():
            pattern = info['pattern']
            if 'func' in info:
                # Use a custom function to determine the replacement
                result_str = re.sub(pattern, info['func'], result_str)
            elif 'exact' in info:
                # Use a fixed replacement
                if re.search(pattern, result_str):
                    self.logger.debug(f"Found calculus pattern: {name}")
                    result_str = re.sub(pattern, info['exact'], result_str)
        
        # Check for decimal approximations of known constants
        # This handles standalone constants like "3.14159" -> "π"
        for symbol, value in self.MATH_CONSTANTS.items():
            # Convert value to string with different precision levels to catch variations
            patterns = [
                f"{value:.2f}",
                f"{value:.3f}",
                f"{value:.4f}",
                f"{value:.5f}",
                f"{value:.6f}"
            ]
            
            # Remove trailing zeros
            patterns = [p.rstrip('0').rstrip('.') for p in patterns]
            
            # Build a pattern that matches any of these variations
            # Use word boundaries to avoid matching substrings of larger numbers
            decimal_pattern = r'(?<!\d)(' + '|'.join(re.escape(p) for p in patterns) + r')(?!\d)'
            
            # Check if pattern exists in the result
            if re.search(decimal_pattern, result_str):
                # Replace decimal with symbol
                result_str = re.sub(decimal_pattern, symbol, result_str)
                self.logger.debug(f"Replaced decimal with symbol: {symbol}")
        
        # Handle large fractions with coefficients: (4159668786720471*x^2)/1125899906842624
        large_fraction_pattern = r'\((\d{10,})\s*\*\s*([^)]+)\)\s*\/\s*(\d{10,})'
        
        def replace_large_fraction(match):
            numerator = int(match.group(1))
            expr = match.group(2)
            denominator = int(match.group(3))
            
            # Calculate the exact ratio
            ratio = numerator / denominator
            
            # Check if ratio approximates a known constant
            for symbol, value in self.MATH_CONSTANTS.items():
                if abs(ratio - value) < 0.0001:
                    self.logger.debug(f"Found large fraction matching {symbol}")
                    return f"{symbol}*{expr}"
            
            # Check for e^2/2 (approximately 3.6945...)
            e_squared_over_2 = 7.3890560989306495 / 2  # Value of e^2/2
            if abs(ratio - e_squared_over_2) < 0.001 and 'x^2' in expr:
                self.logger.debug(f"Recognized e^2*x^2/2 pattern from large fraction with coefficient {ratio}")
                return '(e^2)*x^2/2'
            
            # If expr has a power (like x^n), check if ratio * n is a small integer
            # This helps identify patterns like x^n/n
            power_match = re.match(r'x\^(\d+)', expr)
            if power_match:
                power = int(power_match.group(1))
                product = ratio * power
                
                # If product is close to an integer and that integer divides the power evenly
                if abs(product - round(product)) < 0.0001 and power % round(product) == 0:
                    simplified_power = power // round(product)
                    if simplified_power == 1:
                        return f"x^{power}/{round(product)}"
                    else:
                        return f"x^{simplified_power}/{round(product/simplified_power)}"
            
            # Default decimal approximation with rounding
            rounded = round(ratio, 4)
            coefficient_str = str(rounded).rstrip('0').rstrip('.')
            return f"{coefficient_str}*{expr}"
        
        # Apply large fraction replacement
        result_str = re.sub(large_fraction_pattern, replace_large_fraction, result_str)
        
        # Also handle pattern without parentheses: 4159668786720471*x^2/1125899906842624
        simple_large_fraction_pattern = r'(\d{10,})\s*\*\s*([^/]+)\/(\d{10,})'
        result_str = re.sub(simple_large_fraction_pattern, replace_large_fraction, result_str)
        
        # Handle large fractions without variables: 4159668786720471/1125899906842624
        numeric_fraction_pattern = r'(\d{10,})\s*\/\s*(\d{10,})'
        
        def replace_numeric_fraction(match):
            numerator = int(match.group(1))
            denominator = int(match.group(2))
            ratio = numerator / denominator
            
            # Check if ratio approximates a known constant
            for symbol, value in self.MATH_CONSTANTS.items():
                if abs(ratio - value) < 0.0001:
                    self.logger.debug(f"Found numeric fraction matching {symbol}")
                    return symbol
            
            # Apply general rounding
            rounded = self._smart_round(ratio)
            return str(rounded)
        
        result_str = re.sub(numeric_fraction_pattern, replace_numeric_fraction, result_str)
        
        # General decimal rounding for standalone numbers
        # This handles final cleanup of any remaining decimal numbers
        decimal_pattern = r'(\d+\.\d{5,})'
        
        def round_decimal(match):
            value = float(match.group(1))
            return str(self._smart_round(value))
        
        result_str = re.sub(decimal_pattern, round_decimal, result_str)
        
        return result_str

    def _smart_round(self, value):
        """
        Apply smart rounding to a numerical value.
        Uses different precision depending on the magnitude.
        
        Args:
            value (float): The value to round
            
        Returns:
            str or float: The rounded value
        """
        # Check if value is very close to an integer
        if abs(value - round(value)) < 1e-10:
            return int(round(value))
        
        # Use different precision depending on the magnitude
        abs_value = abs(value)
        
        if abs_value < 0.001:
            # Scientific notation for very small numbers
            return float(f"{value:.2e}")
        elif abs_value < 0.1:
            # More precision for small numbers
            rounded = round(value, 6)
        elif abs_value < 1:
            rounded = round(value, 5)
        elif abs_value < 10:
            rounded = round(value, 4)
        elif abs_value < 100:
            rounded = round(value, 3)
        else:
            rounded = round(value, 1)
        
        # Remove trailing zeros
        result_str = str(rounded)
        if '.' in result_str:
            result_str = result_str.rstrip('0').rstrip('.')
            
        return result_str
        
    def _check_power_pattern(self, match):
        """
        Helper method to check if a coefficient matches a power pattern like x^n/n.
        
        Args:
            match: Regex match object with coefficient and power
            
        Returns:
            str: The simplified expression if pattern matches, or original string
        """
        coefficient = float(match.group(1))
        power = int(match.group(2))
        
        # Check if coefficient * power is close to 1.0 (pattern x^n/n)
        if abs(coefficient * power - 1.0) < 0.001:
            self.logger.debug(f"Found x^n/n pattern: x^{power}/{power}")
            return f"x^{power}/{power}"
        
        # Check for other integer divisors
        for divisor in range(2, 21):  # Check divisors 2 through 20
            if abs(coefficient * divisor - 1.0) < 0.001:
                self.logger.debug(f"Found x^n/m pattern: x^{power}/{divisor}")
                return f"x^{power}/{divisor}"
        
        # No pattern match, return original
        return match.group(0)
