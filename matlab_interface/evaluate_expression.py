import matlab.engine
import logging
import re

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
        
        # Convert log(E, x) to log(x) for natural logarithm
        expression = re.sub(r'log\s*\(\s*E\s*,\s*([^,)]+)\)', r'log(\1)', expression)
        
        # Handle log with different bases, e.g., log(b, x) -> log(x)/log(b)
        expression = re.sub(r'log\s*\(\s*(\d+)\s*,\s*([^,)]+)\)', r'log(\2)/log(\1)', expression)
        
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
        
        # Format log10 as log
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
            # Preprocess the expression
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
                return str(float(result))  # Return as string for consistency

            # Check for symbolic variables
            has_symbols = self.eng.eval("~isempty(symvar(temp_result))", nargout=1)
            self.logger.debug(f"Does the result have symbolic variables? {has_symbols}")

            if has_symbols:
                result_str = self.eng.eval("char(temp_result)", nargout=1)
                self.logger.debug(f"Symbolic result obtained: {result_str}")
                return self._postprocess_result(result_str, is_numeric=False)
            else:
                result = self.eng.eval("double(temp_result)", nargout=1)
                self.logger.debug(f"Double result obtained: {result}")
                return str(float(result))

        except matlab.engine.MatlabExecutionError as me:
            self.logger.error(f"MATLAB Execution Error: {me}", exc_info=True)
            raise ValueError(f"Unexpected Error: {me}") from me
        except Exception as e:
            self.logger.error(f"Unexpected Error: {e}", exc_info=True)
            raise ValueError(f"Unexpected Error: {e}") from e
        finally:
            try:
                self.eng.eval("clear temp_result", nargout=0)
                self.logger.debug("Cleared 'temp_result' from MATLAB workspace.")
            except Exception as e:
                self.logger.warning(f"Error during cleanup: {e}")