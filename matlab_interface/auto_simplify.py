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
        """Configure logging for the AutoSimplify class."""
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

    def simplify_expression(self, expression: str, mode: str = 'basic') -> str:
        """
        Simplify a mathematical expression using different simplification strategies.
        
        Args:
            expression (str): The mathematical expression to simplify.
            mode (str): Simplification mode ('basic', 'full', 'trig', 'rational').
                       - 'basic': Basic algebraic simplification
                       - 'full': Comprehensive simplification
                       - 'trig': Focus on trigonometric simplification
                       - 'rational': Simplify rational expressions
        
        Returns:
            str: The simplified expression.
        """
        self.logger.debug(f"Simplifying expression: '{expression}' using mode: {mode}")
        
        try:
            # Preprocess the expression
            processed_expr = self._preprocess_expression(expression)
            
            # Create MATLAB symbolic expression
            self.eng.eval(f"expr = {processed_expr};", nargout=0)
            
            # Apply simplification based on mode
            simplified = self._apply_simplification(mode)
            
            # Post-process the result
            result = self._postprocess_result(simplified)
            
            self.logger.debug(f"Simplified result: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error during simplification: {e}")
            raise ValueError(f"Simplification error: {str(e)}")
        finally:
            # Clean up workspace
            self._cleanup_workspace()

    def _preprocess_expression(self, expression: str) -> str:
        """
        Preprocess the expression before simplification.
        
        Args:
            expression (str): The input expression.
            
        Returns:
            str: Preprocessed expression.
        """
        # Replace common mathematical notations
        replacements = {
            'ln': 'log',
            'arcsin': 'asin',
            'arccos': 'acos',
            'arctan': 'atan',
            '²': '^2',
            '³': '^3',
            '⁴': '^4',
        }
        
        processed = expression
        for old, new in replacements.items():
            processed = processed.replace(old, new)
            
        # Ensure proper multiplication syntax
        processed = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', processed)
        
        self.logger.debug(f"Preprocessed expression: {processed}")
        return processed

    def _apply_simplification(self, mode: str) -> str:
        """
        Apply the appropriate simplification strategy based on mode.
        
        Args:
            mode (str): Simplification mode.
            
        Returns:
            str: Simplified expression from MATLAB.
        """
        simplification_commands = {
            'basic': "simplify(expr)",
            'full': "simplify(expr, 'Steps', 50)",
            'trig': "simplify(trigsimp(expr))",
            'rational': "simplify(factor(expr))"
        }
        
        command = simplification_commands.get(mode, simplification_commands['basic'])
        self.logger.debug(f"Applying simplification command: {command}")
        
        # Execute simplification in MATLAB
        self.eng.eval(f"result = {command};", nargout=0)
        
        # Get the result as a string
        result = self.eng.eval("char(result)", nargout=1)
        return result

    def _postprocess_result(self, result: str) -> str:
        """
        Post-process the simplified result.
        
        Args:
            result (str): The simplified expression from MATLAB.
            
        Returns:
            str: The post-processed result.
        """
        # Replace MATLAB-specific notations with standard mathematical notations
        # Add replacement for the specific numeric fraction representing 'e'
        replacements = {
            'log': 'ln',                 # Convert back to natural log notation
            'exp(1)': 'e',               # Replace exp(1) with e
            '6121026514868073/2251799813685248': 'e',  # Replace the specific fraction with e
            '.*': '*',                   # Remove element-wise operators
            '.^': '^',                   # Remove element-wise power
            './': '/'                    # Remove element-wise division
        }
        
        processed = result
        for old, new in replacements.items():
            processed = processed.replace(old, new)
            
        self.logger.debug(f"Postprocessed result: {processed}")
        return processed

    def _cleanup_workspace(self):
        """Clean up temporary variables from MATLAB workspace."""
        try:
            self.eng.eval("clear expr result", nargout=0)
            self.logger.debug("Cleaned up MATLAB workspace")
        except Exception as e:
            self.logger.warning(f"Error during workspace cleanup: {e}")

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
