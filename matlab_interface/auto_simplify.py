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
            
            # Check for exponential expressions
            if 'exp(' in expr_str:
                expr_str = re.sub(r'exp\((.*?)\)', r'e^\1', expr_str)
                return expr_str
            
            self.eng.eval(f"syms x; temp_result = {expr_str};", nargout=0)
            self.eng.eval("temp_result = vpa(simplify(temp_result), 4);", nargout=0)
            result = self.eng.eval("char(temp_result)", nargout=1)
            
            if 'e^' in result or 'exp(' in result:
                result = result.replace('exp(', 'e^')
                result = result.rstrip(')')
            
            # Round the result
            rounded_result = self._round_result(result)
            
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
        
        expr = expr.replace('*1.0', '')
        expr = expr.replace('1.0*', '')
        
        expr = expr.replace('log(', 'ln(')
        
        expr = re.sub(r'\b1\*\s*([a-zA-Z])', r'\1', expr)
        
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
            (3.14159265358979, "pi"), (3.14159265358979/2, "pi/2"), 
            (3.14159265358979/3, "pi/3"), (3.14159265358979/4, "pi/4"),
            (3.14159265358979/6, "pi/6")
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
                
        # For other values, return formatted float
        if abs(value) > 1e6 or abs(value) < 1e-6:
            return f"{value:.8e}"  # Scientific notation for very large/small values
        else:
            # Format with appropriate precision avoiding trailing zeros
            result = f"{value:.10f}".rstrip('0').rstrip('.')
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
