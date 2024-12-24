import re
import sympy as sy

class SympyToMatlab:
    def __init__(self):
        import logging
        self.logger = logging.getLogger(__name__)

    def sympy_to_matlab(self, expr):
        """Convert SymPy expression to MATLAB format."""
        self.logger.debug(f"Converting SymPy expression to MATLAB: {expr}")
        
        try:
            # Handle list expressions
            expr = self._handle_list_expression(expr)
            
            # Handle special expression types
            type_handlers = {
                sy.Integral: self._handle_integral,
                sy.Derivative: self._handle_derivative,
                sy.Eq: self._handle_equation,
                sy.Function: self._handle_function
            }
            
            for expr_type, handler in type_handlers.items():
                if isinstance(expr, expr_type):
                    result = handler(expr)
                    self.logger.debug(f"Handled {expr_type.__name__} expression. Result: {result}")
                    return str(result)  # Ensure the result is a string
            
            # For other expressions, convert to string and process
            result = self._process_expression_string(expr)
            self.logger.debug(f"Processed expression string. Result: {result}")
            return str(result)  # Ensure the result is a string
        
        except Exception as e:
            self.logger.error(f"Error in sympy_to_matlab conversion: {e}", exc_info=True)
            raise
    
    def _handle_list_expression(self, expr):
        """Handle list expressions."""
        if isinstance(expr, list):
            if not expr:
                raise ValueError("Empty expression list")
            return expr[0]
        return expr
        
    def _handle_integral(self, expr):
        """Handle integral expressions."""
        self.logger.debug(f"Handling integral expression: {expr}")
        
        # Get the expression being integrated
        integrand = expr.args[0]
        
        # Get the integration variable and limits
        var_info = expr.limits[0]  # Using limits instead of variables
        
        if len(var_info) == 1:
            # Indefinite integral
            var = var_info[0]
            matlab_integral = f"int({self.sympy_to_str(integrand)}, '{var}')"
        else:
            # Definite integral
            var, lower, upper = var_info
            matlab_integral = f"int({self.sympy_to_str(integrand)}, '{var}', {lower}, {upper})"
        
        self.logger.debug(f"Converted integral to MATLAB syntax: {matlab_integral}")
        return matlab_integral
    
    def _handle_derivative(self, expr):
        """Handle derivative expressions."""
        self.logger.debug(f"Handling derivative expression: {expr}")
        
        # Get the expression being differentiated
        func_expr = expr.expr
        
        # Extract the variable and order of differentiation
        var_info = expr.variables[0]
        
        # Check if var_info is a tuple (indicating higher-order derivative)
        if isinstance(var_info, tuple):
            var, order = var_info
        else:
            var = var_info
            order = expr.derivative_count  # Use derivative_count for the order
        
        # Convert the function part
        func_str = self.sympy_to_str(func_expr)
        
        # Create MATLAB derivative expression
        if order == 1:
            matlab_derivative = f"diff({func_str}, '{var}')"
        else:
            matlab_derivative = f"diff({func_str}, '{var}', {order})"
        
        self.logger.debug(f"Converted derivative to MATLAB syntax: {matlab_derivative}")
        return matlab_derivative
    
    def _handle_equation(self, expr):
        """Handle equation expressions."""
        self.logger.debug(f"Handling equation expression: {expr}")
        # Implement equation handling if necessary
        return str(expr)
    
    def _handle_function(self, expr):
        """Handle function expressions like sin, cos, etc."""
        self.logger.debug(f"Handling function expression: {expr}")
        
        func_name = expr.func.__name__
        args = expr.args
        
        # Handle logarithms with different bases
        if func_name == 'log' and len(args) == 2:
            # log(x, base)
            x, base = args
            if base == 10:
                matlab_func = 'log10'
                matlab_args = self.sympy_to_str(x)
                matlab_expression = f"{matlab_func}({matlab_args})"
            elif base == 2:
                matlab_func = 'log2'
                matlab_args = self.sympy_to_str(x)
                matlab_expression = f"{matlab_func}({matlab_args})"
            else:
                # For arbitrary bases, use change of base formula
                matlab_expression = f"log({self.sympy_to_str(x)})/log({self.sympy_to_str(base)})"
            self.logger.debug(f"Converted log with base to MATLAB syntax: {matlab_expression}")
            return matlab_expression
        
        # Existing function mappings with degree-specific trig functions
        function_mappings = {
            'sin': 'sind',
            'cos': 'cosd',
            'tan': 'tand',
            'sind': 'sind',
            'cosd': 'cosd',
            'tand': 'tand',
            'ln': 'log',      # MATLAB uses log for natural logarithm
            'log10': 'log10', # Ensure log10 is mapped correctly
            'log2': 'log2',   # Handle base-2 logs if needed
            # Add more mappings as needed
        }
        
        matlab_func = function_mappings.get(func_name, func_name)
        matlab_args = ', '.join([self.sympy_to_str(arg) for arg in args])
        matlab_expression = f"{matlab_func}({matlab_args})"
        self.logger.debug(f"Converted function to MATLAB syntax: {matlab_expression}")
        return matlab_expression
    
    def sympy_to_str(self, expr):
        """Convert SymPy expression to string."""
        return str(expr)
    
    def _process_expression_string(self, expr_str):
        """Process a general expression string."""
        self.logger.debug(f"Processing general expression string: {expr_str}")
        # Implement processing logic (e.g., handling operators, functions)
        return expr_str
        
    def _is_degree_mode(self):
        """Determine if the calculator is in degree mode."""
        # This method should determine if the calculator is set to degree mode
        # For now, we'll assume radian mode. You can modify this based on your application state.
        return False
        