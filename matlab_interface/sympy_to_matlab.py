import re
import sympy as sy
import logging

class SympyToMatlab:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def sympy_to_matlab(self, expr):
        """Convert SymPy expression to MATLAB format."""
        self.logger.debug(f"Converting SymPy expression to MATLAB: {expr}")
        
        if isinstance(expr, str):
            return expr
        
        # Handle derivatives
        if expr.is_Derivative:
            return self._handle_derivative(expr)
        
        # Convert the expression to a string
        expr_str = str(expr)
        
        # Replace Python-style power operator with MATLAB-style
        expr_str = expr_str.replace('**', '^')
        
        # Ensure proper multiplication syntax
        expr_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr_str)
        
        self.logger.debug(f"Processed expression string. Result: {expr_str}")
        return expr_str

    def _handle_list_expression(self, expr):
        """
        Handle list or tuple expressions, converting them to MATLAB array syntax.

        Args:
            expr (sympy.Expr or list or tuple): The expression or list to handle.

        Returns:
            str: MATLAB-formatted string.
        """
        if isinstance(expr, (list, tuple)):
            matlab_list = '[' + ', '.join([self.sympy_to_matlab(e) for e in expr]) + ']'
            self.logger.debug(f"Converted list expression to MATLAB array: {matlab_list}")
            return matlab_list
        else:
            return self._process_expression_string(expr)

    def _handle_integral(self, expr):
        """Handle integral expressions."""
        self.logger.debug(f"Handling integral expression: {expr}")
        
        # Get the expression being integrated
        integrand = expr.args[0]
        
        # Recursively handle nested integrals
        if isinstance(integrand, sy.Integral):
            integrand_str = self.sympy_to_matlab(integrand)
        else:
            integrand_str = self.sympy_to_str(integrand)
        
        # Get the integration variable and limits
        var_info = expr.limits[0]
        
        if len(var_info) == 1:
            # Indefinite integral
            var = var_info[0]
            matlab_integral = f"int({integrand_str}, {var})"
        else:
            # Definite integral
            var, lower, upper = var_info
            lower_str = self.sympy_to_matlab(lower)
            upper_str = self.sympy_to_matlab(upper)
            matlab_integral = f"int({integrand_str}, {var}, {lower_str}, {upper_str})"
        
        self.logger.debug(f"Converted integral to MATLAB syntax: {matlab_integral}")
        return matlab_integral

    def sympy_to_str(self, expr):
        """Convert SymPy expression to string, handling constants."""
        expr_str = str(expr)
        # Replace 'E' with 'exp(1)' for MATLAB compatibility
        expr_str = expr_str.replace('E', 'exp(1)')
        return expr_str

    def _process_expression_string(self, expr):
        """Process a general expression string."""
        self.logger.debug(f"Processing general expression: {expr}")
        
        # Convert the SymPy expression to a string
        expr_str = self.sympy_to_str(expr)
        
        # Define a dictionary for replacements
        replacements = {
            '**': '^',  # Replace exponentiation
            # Ensure other operators are correctly handled if needed
        }
        
        # Apply replacements
        for sympy_op, matlab_op in replacements.items():
            expr_str = expr_str.replace(sympy_op, matlab_op)
        
        self.logger.debug(f"After replacements: {expr_str}")
        return expr_str

    def _handle_derivative(self, expr):
        """Handle derivative expressions."""
        self.logger.debug(f"Handling derivative expression: {expr}")
        
        # Get the expression being differentiated
        func_expr = expr.expr
        
        # Extract the variable and order of differentiation
        var = expr.variables[0]
        order = expr.derivative_count if hasattr(expr, 'derivative_count') else 1
        
        # Convert the function part
        func_str = self.sympy_to_matlab(func_expr)
        
        # Create MATLAB derivative expression using diff instead of Derivative
        if order == 1:
            matlab_derivative = f"diff({func_str}, {var})"
        else:
            matlab_derivative = f"diff({func_str}, {var}, {order})"
        
        self.logger.debug(f"Converted derivative to MATLAB syntax: {matlab_derivative}")
        return matlab_derivative

    def _handle_equation(self, expr):
        """Handle equation expressions."""
        self.logger.debug(f"Handling equation expression: {expr}")
        # Implement equation handling if necessary
        # For example: lhs = rhs
        lhs, rhs = expr.lhs, expr.rhs
        matlab_eq = f"{self.sympy_to_matlab(lhs)} == {self.sympy_to_matlab(rhs)}"
        return matlab_eq

    def _handle_function(self, expr):
        """Handle function expressions like sin, cos, etc."""
        self.logger.debug(f"Handling function expression: {expr}")
        
        func_name = expr.func.__name__
        args = expr.args
        
        # Handle special functions if necessary
        function_mappings = {
            'sin': 'sin',
            'cos': 'cos',
            'tan': 'tan',
            'csc': 'csc',
            'sec': 'sec',
            'cot': 'cot',
            'arcsin': 'asin',
            'arccos': 'acos',
            'arctan': 'atan',
            'ln': 'log',      # MATLAB uses log for natural logarithm
            'log10': 'log10',
            'log2': 'log2',
            # Add more mappings as needed
        }
        
        matlab_func = function_mappings.get(func_name, func_name)
        matlab_args = ', '.join([self.sympy_to_matlab(arg) for arg in args])
        
        # Handle absolute value separately if needed
        if func_name == 'abs':
            matlab_expression = f"abs({matlab_args})"
        else:
            matlab_expression = f"{matlab_func}({matlab_args})"
        
        self.logger.debug(f"Converted function to MATLAB syntax: {matlab_expression}")
        return matlab_expression
        