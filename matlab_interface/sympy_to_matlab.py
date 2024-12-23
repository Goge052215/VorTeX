import re
import sympy as sy

class SympyToMatlab:
    def __init__(self):
        # Initialize logger if needed
        import logging
        self.logger = logging.getLogger(__name__)

    def sympy_to_matlab(self, expr):
        """Convert SymPy expression to MATLAB format."""
        self.logger.debug(f"Converting SymPy expression to MATLAB: {expr}")
        
        try:
            # Handle empty or list expressions
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
                    return result
            
            # Convert to string and apply transformations
            result = self._process_expression_string(expr)
            self.logger.debug(f"Processed expression string. Result: {result}")
            return result
        
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
        try:
            func = self.sympy_to_matlab(expr.function)
            var = expr.variables[0]

            # Handle logarithm conversions
            func = self._convert_logarithms(func)

            if len(expr.limits) == 0:
                result = f"int({func}, {var})"
                self.logger.debug(f"Created indefinite integral: {result}")
                return result

            if len(expr.limits) == 1:
                if len(expr.limits[0]) == 3:
                    a, b = expr.limits[0][1], expr.limits[0][2]
                    result = f"int({func}, {var}, {a}, {b})"
                    self.logger.debug(f"Created definite integral: {result}")
                    return result
                result = f"int({func}, {var})"
                self.logger.debug(f"Created integral with variable: {result}")
                return result

            self.logger.error("Unsupported integral type encountered")
            raise ValueError("Unsupported integral type")

        except Exception as e:
            self.logger.error(f"Error handling integral: {e}")
            raise
        
    def _handle_derivative(self, expr):
        """Handle derivative expressions."""
        self.logger.debug(f"Handling derivative expression: {expr}")
        try:
            # Get the function being differentiated
            func = self.sympy_to_matlab(expr.expr)

            # Get the variable and order of differentiation
            var = str(expr.variables[0])

            # Handle order of derivative
            if hasattr(expr, 'order'):
                order = expr.order
            else:
                # Count the number of times the same variable appears
                order = sum(1 for v in expr.variables if str(v) == var)

            # Convert logarithms if present
            func = self._convert_logarithms(func)

            # Create MATLAB derivative command
            if order > 1:
                result = f"diff({func}, {var}, {order})"
            else:
                result = f"diff({func}, {var})"

            self.logger.debug(f"Created derivative expression: {result}")
            return result

        except Exception as e:
            self.logger.error(f"Error handling derivative: {e}", exc_info=True)
            raise
        
    def _handle_equation(self, expr):
        """Handle equation expressions."""
        self.logger.debug(f"Handling equation expression: {expr}")
        try:
            lhs = self.sympy_to_matlab(expr.lhs)
            rhs = self.sympy_to_matlab(expr.rhs)
            result = f"{lhs} == {rhs}"
            self.logger.debug(f"Created equation: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Error handling equation: {e}")
            raise
        
    def _handle_function(self, expr):
        """Handle function expressions."""
        self.logger.debug(f"Handling function expression: {expr}")
        try:
            func_name = expr.func.__name__
            args = [self.sympy_to_matlab(arg) for arg in expr.args]

            function_handlers = {
                'sin': lambda args: f"sind({args[0]})" if self._is_degree_mode() else f"sin({args[0]})",
                'cos': lambda args: f"cosd({args[0]})" if self._is_degree_mode() else f"cos({args[0]})",
                'tan': lambda args: f"tand({args[0]})" if self._is_degree_mode() else f"tan({args[0]})",
                'ln': lambda args: f"log({args[0]})",  # MATLAB's natural log
                'sqrt': lambda args: f"sqrt({args[0]})",
                'Abs': lambda args: f"abs({args[0]})",
                'factorial': lambda args: f"factorial({args[0]})",
                'nchoosek': lambda args: f"nchoosek({args[0]}, {args[1]})"
            }

            if func_name in function_handlers:
                result = function_handlers[func_name](args)
                self.logger.debug(f"Handled special function {func_name}: {result}")
                return result

            result = f"{func_name}({', '.join(args)})"
            self.logger.debug(f"Handled generic function {func_name}: {result}")
            return result

        except Exception as e:
            self.logger.error(f"Error handling function: {e}")
            raise
        
    def _process_expression_string(self, expr):
        """Process and convert expression string to MATLAB format."""
        expr_str = str(expr)

        # Replace LaTeX symbols
        expr_str = self._replace_latex_symbols(expr_str)

        # Convert logarithms
        expr_str = self._convert_logarithms(expr_str)

        # Apply function replacements
        expr_str = self._apply_function_replacements(expr_str)

        # Handle integrals and derivatives
        expr_str = self._handle_integral_derivative_patterns(expr_str)

        # Process operators and special cases
        expr_str = self._process_operators(expr_str)

        # Handle combinations and permutations
        expr_str = self._handle_combinations_permutations(expr_str)

        return expr_str
        
    def _replace_latex_symbols(self, expr_str):
        """Replace LaTeX-specific symbols with MATLAB equivalents."""
        replacements = {
            r'\pi': 'pi',
            r'\e': 'exp(1)',
            r'\alpha': 'alpha',  # Add more as needed
            r'\beta': 'beta'
            # Add additional LaTeX symbol replacements here
        }

        for latex_sym, matlab_sym in replacements.items():
            expr_str = expr_str.replace(latex_sym, matlab_sym)

        return expr_str
        
    def _convert_logarithms(self, expr_str):
        """Convert different types of logarithms."""
        if 'log(x, E)' in expr_str or 'log(x)' in expr_str:
            expr_str = expr_str.replace('log(x, E)', 'log(x)')
            expr_str = expr_str.replace('log(x)', 'log(x)')
        elif 'log(x, 10)' in expr_str:
            expr_str = expr_str.replace('log(x, 10)', 'log10(x)')
        return expr_str
        
    def _apply_function_replacements(self, expr_str):
        """Apply function replacements using the replacement dictionary."""
        replacements = {
            'ln': 'log', 'log10': 'log10', 'log2': 'log2',
            'sin': 'sin', 'cos': 'cos', 'tan': 'tan',
            'asin': 'asin', 'acos': 'acos', 'atan': 'atan',
            'log': 'log', 'exp': 'exp', 'sqrt': 'sqrt',
            'factorial': 'factorial', '**': '^', 'pi': 'pi',
            'E': 'exp(1)', 'Abs': 'abs', 'Derivative': 'diff'
        }

        for sympy_func, matlab_func in replacements.items():
            expr_str = expr_str.replace(sympy_func, matlab_func)
        return expr_str
        
    def _handle_integral_derivative_patterns(self, expr_str):
        """Handle integral and derivative patterns."""
        patterns = {
            r'Integral\((.*?), \((.*?), (.*?), (.*?)\)\)': r'integral(@(x) \1, \3, \4)',
            r'Integral\((.*?), (.*?)\)': r'int(\1, \2)',
            r'Function\((.*?)\)': r'\1'
        }

        for pattern, replacement in patterns.items():
            expr_str = re.sub(pattern, replacement, expr_str)
        return expr_str
        
    def _process_operators(self, expr_str):
        """Process mathematical operators."""
        expr_str = expr_str.replace('**', '.^')
        expr_str = expr_str.replace('/', './').replace('*', '.*')
        expr_str = expr_str.replace('.*1', '').replace('.*0', '0')
        return expr_str
        
    def _handle_combinations_permutations(self, expr_str):
        """Handle combinations and permutations patterns."""
        patterns = {
            r'\\binom\s*\{\s*(\d+)\s*\}\s*\{\s*(\d+)\s*\}': r'nchoosek(\1, \2)',
            r'\\choose\s*\{\s*(\d+)\s*\}\s*\{\s*(\d+)\s*\}': r'nchoosek(\1, \2)',
            r'\bnCr\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)': r'nchoosek(\1, \2)',
            r'\bnPr\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)': r'nchoosek(\1, \2)*factorial(\2)'
        }

        for pattern, replacement in patterns.items():
            expr_str = re.sub(pattern, replacement, expr_str, flags=re.IGNORECASE)
        return expr_str
        
    def _is_degree_mode(self):
        """Determine if the calculator is in degree mode."""
        # Implement a method to check if the current mode is degrees.
        # This might require passing the mode information from CalculatorApp.
        # For simplicity, assume radian mode here.
        # You need to adjust this based on your actual implementation.
        return False  # Change accordingly
        