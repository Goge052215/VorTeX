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
            
            # Define 'e' as the mathematical constant exp(1)
            self.eng.eval("e = exp(1);", nargout=0)
            
            self.logger.debug("Initialized symbolic variables in MATLAB workspace")
        except Exception as e:
            self.logger.error(f"Error initializing symbolic variables: {e}")
            raise

    def simplify_expression(self, expr_str):
        """
        Simplify a mathematical expression using MATLAB's symbolic toolbox.
        
        Args:
            expr_str (str): The expression to simplify.
            
        Returns:
            str: The simplified expression.
        """
        try:
            self.logger.debug(f"Simplifying expression: '{expr_str}'")
            
            # Assign the expression to MATLAB workspace
            self.eng.workspace['expr'] = expr_str
            
            # Define a sequence of simplification commands
            simplification_commands = [
                "expr = str2sym(expr);",             # Convert string to symbolic
                "expr = vpa(expr, 'Digits', 10);",   # Set precision to prevent decimal conversion
                "expr = simplify(expr, 'Steps', 50);", # Simplify with specified steps
                # Keep expressions in symbolic form
                "expr = sym(expr);",
                # Convert any remaining decimals to fractions or symbolic constants
                "expr = vpa(expr, 'Digits', 10);",
                "expr = simplify(expr, 'IgnoreAnalyticConstraints', true);",
                # Final simplification
                "expr = simplify(expr);",
            ]
            
            # Execute each simplification command
            for cmd in simplification_commands:
                self.logger.debug(f"Executing MATLAB command: {cmd}")
                self.eng.eval(cmd, nargout=0)
            
            # Get the result as a single expression
            result = self.eng.eval("char(expr)", nargout=1)
            self.logger.debug(f"Raw MATLAB result: {result}")
            
            # Clean up the result
            result = self._clean_expression(result)
            self.logger.debug(f"Simplified Result: '{result}'")
            
            return result
                
        except Exception as e:
            self.logger.error(f"Error during simplification: {e}")
            return expr_str
        
        finally:
            # Clean up workspace
            self.eng.eval("clear expr", nargout=0)

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
        
        # Remove unnecessary multiplications by 1.0
        expr = expr.replace('*1.0', '')
        expr = expr.replace('1.0*', '')
        
        # Convert 'exp(x)' to 'e^x' for display
        expr = re.sub(r'exp\((.*?)\)', lambda m: f'e^{m.group(1)}', expr).rstrip(')')
        
        # Replace 'log' with 'ln' for natural logarithm
        expr = expr.replace('log(', 'ln(')
        
        # Ensure multiplication is clearly formatted
        expr = re.sub(r'\b1\*\s*([a-zA-Z])', r'\1', expr)
        
        self.logger.debug(f"Final cleaned expression: '{expr}'")
        
        return expr

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
