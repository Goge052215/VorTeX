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
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)

    def _init_matlab_symbolic_vars(self):
        """Initialize common symbolic variables in MATLAB workspace."""
        try:
            # Initialize common variables (excluding 'e')
            self.eng.eval("syms x y z t a b c n real", nargout=0)
            
            # Define 'e' as the mathematical constant exp(1)
            self.eng.eval("e = exp(1);", nargout=0)
            
            # Optionally, define 'pi' as symbolic if needed
            # MATLAB already has 'pi' defined, so this may be unnecessary
            # self.eng.eval("syms pi", nargout=0)
            
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
        self.logger.debug(f"Simplifying expression: '{expr_str}'")
        
        try:
            # Convert the expression to a symbolic form
            self.eng.workspace['expr'] = expr_str
            
            # Try different simplification methods in sequence
            simplification_commands = [
                "simplify(expr, 'Steps', 50)",  # Basic simplification
                "simplify(collect(expr))",       # Collect similar terms
                "simplify(factor(expr))",        # Try factoring
                "simplify(combine(expr))",       # Combine terms
                "vpa(expr, 10)"                  # Variable precision arithmetic if needed
            ]
            
            for cmd in simplification_commands:
                self.logger.debug(f"Applying simplification command: {cmd}")
                self.eng.eval(f"result = {cmd};", nargout=0)
                
                # Get the current result
                current_result = self.eng.eval("char(result)", nargout=1)
                
                # Try to convert fractions to decimals if they're too complex
                if '/' in current_result:
                    numerator = self.eng.eval("num2str(double(numden(result)), 15)", nargout=1)
                    if not ('e' in numerator.lower() or 'i' in numerator.lower()):
                        self.eng.eval("result = vpa(result, 10);", nargout=0)
                        current_result = self.eng.eval("char(result)", nargout=1)
                
                # Update if simpler
                if len(current_result) < len(expr_str):
                    expr_str = current_result
            
            # Final cleanup
            expr_str = self._clean_expression(expr_str)
            
            self.logger.debug(f"Simplified result: {expr_str}")
            return expr_str
            
        except Exception as e:
            self.logger.error(f"Error during simplification: {e}")
            return expr_str
            
        finally:
            # Clean up workspace
            self.eng.eval("clear expr result", nargout=0)

    def _clean_expression(self, expr_str):
        """
        Clean up the expression by removing unnecessary symbols and improving readability.
        
        Args:
            expr_str (str): The expression to clean.
            
        Returns:
            str: The cleaned expression.
        """
        # Convert 'log' to 'ln'
        expr_str = expr_str.replace('log', 'ln')

        # Remove multiplication signs in specific cases
        patterns = [
            (r'(\d+)\*([a-zA-Z])', r'\1\2'),           # 2*x -> 2x
            (r'([a-zA-Z])\*(\d+)', r'\1\2'),           # x*2 -> x2
            (r'([a-zA-Z])\*([a-zA-Z])', r'\1\2'),      # x*y -> xy
            (r'\)\*([a-zA-Z])', r')\1'),               # )*x -> )x
            (r'\)\*(\d+)', r')\1'),                    # )*2 -> )2
            (r'([a-zA-Z])\*\(', r'\1('),              # x*( -> x(
            (r'(\d+)\*\(', r'\1('),                   # 2*( -> 2(
            (r'\*\*', '^'),                           # ** -> ^
        ]

        for pattern, replacement in patterns:
            expr_str = re.sub(pattern, replacement, expr_str)

        # Clean up whitespace
        expr_str = re.sub(r'\s+', ' ', expr_str).strip()

        # Handle special cases for negative numbers
        expr_str = re.sub(r'(\d+)\*-', r'\1(-', expr_str)

        return expr_str

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
