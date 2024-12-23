import matlab.engine
import logging

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

    def evaluate_matlab_expression(self, matlab_expression):
        """
        Evaluate a MATLAB expression and return the result.
        
        Args:
            matlab_expression (str): The MATLAB expression to evaluate.
        
        Returns:
            str or float or list: The evaluated result as a formatted string, float, or list.
        
        Raises:
            ValueError: If MATLAB execution fails or the expression is invalid.
        """
        self.logger.debug(f"Starting evaluation of MATLAB expression: '{matlab_expression}'")
        try:
            # Execute the MATLAB expression and assign it to temp_result
            self.eng.eval(f"temp_result = {matlab_expression};", nargout=0)
            self.logger.info("Expression executed successfully in MATLAB.")

            # Check if the result is numeric
            is_numeric = self.eng.eval("isnumeric(temp_result)", nargout=1)
            self.logger.debug(f"Is the result numeric? {is_numeric}")

            if is_numeric:
                # Directly retrieve the numeric result
                result = self.eng.eval("temp_result", nargout=1)
                self.logger.debug(f"Numeric result obtained: {result}")
                return result

            # If not numeric, check for symbolic variables
            has_symbols = self.eng.eval("~isempty(symvar(temp_result))", nargout=1)
            self.logger.debug(f"Does the result have symbolic variables? {has_symbols}")

            if has_symbols:
                # Retrieve the symbolic result as a string
                result_str = self.eng.eval("char(temp_result)", nargout=1)
                self.logger.debug(f"Symbolic result obtained: {result_str}")
                return self._format_result_string(result_str)
            else:
                # Retrieve the result as a double if no symbols are present
                result = self.eng.eval("double(temp_result)", nargout=1)
                self.logger.debug(f"Double result obtained: {result}")
                return result

        except matlab.engine.MatlabExecutionError as me:
            self.logger.error(f"MATLAB Execution Error: {me}", exc_info=True)
            raise ValueError(f"MATLAB Error: {me}") from me
        except Exception as e:
            self.logger.error(f"Unexpected Error: {e}", exc_info=True)
            raise ValueError(f"Unexpected Error: {e}") from e
        finally:
            # Clear the temporary result from MATLAB workspace
            try:
                self.eng.eval("clear temp_result", nargout=0)
                self.logger.debug("Cleared 'temp_result' from MATLAB workspace.")
            except matlab.engine.MatlabExecutionError as me:
                self.logger.warning(f"Failed to clear 'temp_result': {me}")
            except Exception as e:
                self.logger.warning(f"Unexpected error during cleanup: {e}")

    def _format_result_string(self, result_str):
        """
        Format the MATLAB symbolic result string for user-friendly display.
        
        Args:
            result_str (str): The symbolic result string from MATLAB.
        
        Returns:
            str: The formatted result string.
        """
        self.logger.debug("Formatting the result string.")
        # Replace 'log(' with 'ln(' for natural logarithm display
        # Also replace MATLAB's power and multiplication notations with standard ones
        formatted = result_str.replace('log(', 'ln(').replace('.^', '^').replace('.*', '*')
        self.logger.debug(f"Formatted string: '{formatted}'")
        return formatted