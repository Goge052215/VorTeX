import re
import logging
from sympy import sympify, N, pi, limit, Symbol, Integral, diff
from sympy.abc import x, y
import sympy

class SympyCalculation:
    """Class to handle SymPy calculations and expression parsing."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.math_dict = {
            'sin': sympy.sin,
            'cos': sympy.cos,
            'tan': sympy.tan,
            'asin': sympy.asin,
            'acos': sympy.acos,
            'atan': sympy.atan,
            'log': sympy.log,
            'ln': sympy.log,
            'exp': sympy.exp,
            'sqrt': sympy.sqrt,
            'factorial': sympy.factorial,
            'limit': limit,
            'Integral': Integral,
            'diff': diff,
            'binomial': sympy.binomial,
            'x': x,
            'y': y,
            'z': Symbol('z'),
        }

    def parse_expression(self, expression: str) -> str:
        """Parse and prepare the expression for SymPy evaluation."""
        try:
            self.logger.debug(f"Parsing expression: {expression}")
            
            # Handle harmonic series
            harmonic_pattern = r'sum\s*\(\s*(\d+)\s*to\s*(?:inf|Inf|(\d+))\s*\)\s*1/([a-zA-Z])'
            def replace_harmonic(match):
                start = int(match.group(1))
                end = match.group(2)  # Will be None for inf/Inf
                var = match.group(3)
                
                if end is None:  # Infinite series
                    return "oo"  # SymPy's infinity symbol
                else:  # Finite series
                    if start == 1:
                        # For sum from 1 to n, use harmonic number formula
                        return f"log({end}) + 0.57721566490153286060"
                    else:
                        # For sum from k to n
                        return f"log({end}) - log({start-1})"

            if re.search(harmonic_pattern, expression):
                expression = re.sub(harmonic_pattern, replace_harmonic, expression)
                return expression
            
            # Handle product expressions
            prod_pattern = r'prod\s*\(\s*(\d+)\s*to\s*(?:inf|Inf|(\d+))\s*\)\s*([^\n]+)'
            def replace_prod(match):
                start = match.group(1)
                end = match.group(2)  # Will be None for inf/Inf
                expr_part = match.group(3)
                
                if end is None:
                    return f"Product({expr_part}, (x, {start}, oo))"
                else:
                    return f"Product({expr_part}, (x, {start}, {end}))"

            expression = re.sub(prod_pattern, replace_prod, expression)
            
            if 'C' in expression:
                expression = self._parse_combination(expression)
            
            elif 'P' in expression:
                expression = self._parse_permutation(expression)
            
            # Add limit handling
            elif expression.startswith('lim'):
                expression = self._parse_limit(expression)
            
            # Handle derivatives
            elif any(x in expression for x in ['d/dx', 'd/dy', 'diff(']):
                expression = self._parse_derivative(expression)
            
            # Handle integral expressions
            elif expression.startswith('int'):
                expression = self._parse_integral(expression)
            
            # Replace degree symbol and pi
            expression = expression.replace('°', '*pi/180')
            expression = expression.replace('pi', 'sympy.pi')
            
            # Handle power operator
            expression = expression.replace('^', '**')
            
            self.logger.debug(f"Parsed expression: {expression}")
            return expression
            
        except Exception as e:
            self.logger.error(f"Error parsing expression: {e}")
            raise

    def _parse_combination(self, expression: str) -> str:
        """Parse combination expressions (nCr format)."""
        try:
            # Match pattern like "5C2" or "10C3"
            match = re.search(r'(\d+)C(\d+)', expression)
            if match:
                n, r = match.groups()
                return expression.replace(f"{n}C{r}", f"binomial({n}, {r})")
            return expression
        except Exception as e:
            self.logger.error(f"Error parsing combination: {e}")
            raise ValueError(f"Invalid combination format: {expression}")

    def _parse_permutation(self, expression: str) -> str:
        """Parse permutation expressions (nPr format)."""
        try:
            # Match pattern like "5P2" or "10P3"
            match = re.search(r'(\d+)P(\d+)', expression)
            if match:
                n, r = match.groups()
                # P(n,r) = n!/(n-r)!
                return expression.replace(f"{n}P{r}", f"factorial({n})/factorial({n}-{r})")
            return expression
        except Exception as e:
            self.logger.error(f"Error parsing permutation: {e}")
            raise ValueError(f"Invalid permutation format: {expression}")

    def _parse_integral(self, expression: str) -> str:
        """Parse integral expressions."""
        try:
            # First clean up any extra spaces
            expression = expression.strip()
            
            # Remove 'int' and 'dx' and get the integrand
            integrand = expression.replace('int', '', 1).replace('dx', '', 1).strip()
            
            self.logger.debug(f"Extracted integrand: {integrand}")
            
            # Check for definite integral limits
            if integrand.startswith('('):
                # Extract limits and integrand for definite integral
                close_paren = integrand.find(')')
                if close_paren == -1:
                    raise ValueError("Missing closing parenthesis in definite integral")
                
                limits = integrand[1:close_paren].split(',')
                integrand = integrand[close_paren + 1:].strip()
                
                if len(limits) == 2:
                    if not integrand:
                        raise ValueError("Missing integrand in definite integral")
                    return f"Integral({integrand}, (x, {limits[0].strip()}, {limits[1].strip()}))"
            
            # For indefinite integral
            if not integrand:
                raise ValueError("Missing integrand in integral")
            
            # Replace ln with log for SymPy but preserve the argument
            if 'ln(' in integrand:
                integrand = integrand.replace('ln(', 'log(')
            
            self.logger.debug(f"Final integral expression: Integral({integrand}, x)")
            return f"Integral({integrand}, x)"
            
        except Exception as e:
            self.logger.error(f"Error parsing integral: {e}")
            raise ValueError(f"Invalid integral format: {str(e)}")

    def _parse_derivative(self, expression: str) -> str:
        """Parse derivative expressions."""
        try:
            # Handle d/dx format
            if 'd/dx' in expression:
                # Match both d/dx(f) and d/dx f formats
                match = re.search(r'd/dx\((.*?)\)', expression) or re.search(r'd/dx\s*(.*)', expression)
                if match:
                    func = match.group(1).strip()
                    # Handle higher order derivatives
                    order = expression.count('d/dx')
                    return f"diff({func}, x, {order})" if order > 1 else f"diff({func}, x)"
                
            # Handle d/dy format
            elif 'd/dy' in expression:
                match = re.search(r'd/dy\((.*?)\)', expression) or re.search(r'd/dy\s*(.*)', expression)
                if match:
                    func = match.group(1).strip()
                    order = expression.count('d/dy')
                    return f"diff({func}, y, {order})" if order > 1 else f"diff({func}, y)"
                    
            # Handle diff() format
            elif expression.startswith('diff('):
                return expression
                
            self.logger.debug(f"No valid derivative format found in: {expression}")
            return expression
            
        except Exception as e:
            self.logger.error(f"Error parsing derivative: {e}")
            raise ValueError(f"Error parsing derivative: {e}")

    def _parse_limit(self, expression: str) -> str:
        """Parse limit expressions."""
        try:
            # Remove 'lim' and clean up spaces
            expression = expression.strip()[3:].strip()
            
            # Handle lim(x to a) format
            if expression.startswith('('):
                # Remove outer parentheses
                expression = expression[1:].strip()
                if expression.endswith(')'):
                    expression = expression[:-1].strip()
            else:
                raise ValueError("Limit expression must be in format 'lim(x to a)'")
            
            # Split by 'to' keyword
            parts = expression.split(' to ')
            if len(parts) != 2:
                raise ValueError("Missing 'to' keyword in limit expression")
            
            var_approach = parts[0].strip()
            remaining = parts[1].strip()
            
            # Find the first space or closing parenthesis after the approach value
            space_idx = remaining.find(' ')
            paren_idx = remaining.find(')')
            
            # Use the first occurring delimiter
            if space_idx == -1 and paren_idx == -1:
                raise ValueError("Missing function in limit expression")
            elif space_idx == -1:
                split_idx = paren_idx
            elif paren_idx == -1:
                split_idx = space_idx
            else:
                split_idx = min(space_idx, paren_idx)
            
            approach_val = remaining[:split_idx].strip()
            function = remaining[split_idx:].strip()
            
            # Remove any remaining parentheses from the function
            function = function.strip('()')
            
            return f"limit({function}, {var_approach}, {approach_val})"
            
        except Exception as e:
            self.logger.error(f"Error parsing limit: {e}")
            raise ValueError(f"Invalid limit format: {str(e)}")

    def evaluate(self, expression: str, angle_mode: str = 'Radian') -> str:
        """Evaluate the SymPy expression and return the result."""
        try:
            self.logger.debug(f"Evaluating expression: {expression}")
            
            # Parse the expression
            parsed_expr = self.parse_expression(expression)
            
            # Evaluate using SymPy
            result = sympify(parsed_expr, locals=self.math_dict)
            
            # Further evaluate integrals and other symbolic expressions
            if isinstance(result, sympy.Integral):
                result = result.doit()
            elif isinstance(result, sympy.Derivative):
                result = result.doit()
            
            # Try to get numeric result
            try:
                # For purely numeric results
                if result.is_number:
                    numeric_result = float(N(result))
                    if angle_mode == 'Degree' and any(trig in expression for trig in ['asin', 'acos', 'atan']):
                        numeric_result = float(numeric_result * 180 / float(N(pi)))
                    result = str(round(numeric_result, 8))
                else:
                    # For symbolic results, try to simplify
                    result = sympy.simplify(result)
                    result = str(result)
            except:
                result = str(result)
            
            # Clean up display
            result = self._clean_display(result, angle_mode)
            
            self.logger.debug(f"Evaluation result: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating expression: {e}")
            raise

    def _clean_display(self, result: str, angle_mode: str) -> str:
        """Clean up the display format of the result."""
        try:
            # Replace mathematical notations
            result = result.replace('log(', 'ln(')
            result = result.replace('**', '^')
            result = result.replace('exp(1)', 'e')
            
            # Handle trigonometric functions in degree mode
            if angle_mode == 'Degree':
                result = result.replace('sin(', 'sin(')
                result = result.replace('cos(', 'cos(')
                result = result.replace('tan(', 'tan(')
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error cleaning display: {e}")
            return result
