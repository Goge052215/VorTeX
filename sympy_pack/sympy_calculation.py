'''
Copyright (c) 2025 George Huang. All Rights Reserved.

This file is part of the VorTeX Calculator project.
Component: sympy calculation - Sympy Evaluator

This file and its contents are protected under international copyright laws.
No part of this file may be reproduced, distributed, or transmitted in any form
or by any means, including photocopying, recording, or other electronic or
mechanical methods, without the prior written permission of the copyright owner.

PROPRIETARY AND CONFIDENTIAL
This file contains proprietary and confidential information that implements
core MATLAB interface functionality. Unauthorized copying, distribution, or use 
of this file, via any medium, is strictly prohibited.

LICENSE RESTRICTIONS
- Commercial use is strictly prohibited without explicit written permission
- Modifications to this file are not permitted
- Distribution or sharing of this file is not permitted
- Private use must maintain all copyright and license notices
- Any attempt to reverse engineer the MATLAB interface is prohibited

Version: 1.0.2
Last Updated: 2025.1
'''

import re
import sys
import logging
from sympy import sympify, N, pi, limit, Symbol, Integral, diff
from sympy.abc import x, y
import sympy
import psutil
from typing import Optional
from functools import wraps

IS_WINDOWS = sys.platform in ["win32", "win64"]

try:
    import resource
    RESOURCE_AVAILABLE = True
except ImportError:
    RESOURCE_AVAILABLE = False

class SympyCalculation:
    """Class to handle SymPy calculations and expression parsing."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.max_memory = 512 * 1024 * 1024
        self.max_output_length = 10000
        self._process: Optional[psutil.Process] = None
        try:
            self._process = psutil.Process()
        except ImportError:
            self.logger.warning("psutil not available, memory management disabled")
        
        self.math_dict = {
            'sin': sympy.sin,
            'cos': sympy.cos,
            'tan': sympy.tan,
            'sec': sympy.sec,
            'csc': sympy.csc,
            'cot': sympy.cot,
            'asin': sympy.asin,
            'acos': sympy.acos,
            'atan': sympy.atan,
            'arcsec': sympy.asec,
            'arccsc': sympy.acsc,
            'arccot': sympy.acot,
            'sinh': sympy.sinh,
            'cosh': sympy.cosh,
            'tanh': sympy.tanh,
            'sech': sympy.sech,
            'csch': sympy.csch,
            'coth': sympy.coth,
            'asinh': sympy.asinh,
            'acosh': sympy.acosh,
            'atanh': sympy.atanh,
            'arcsinh': sympy.asinh,
            'arccosh': sympy.acosh,
            'arctanh': sympy.atanh,
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

    def memory_safe(func):
        """Decorator to enforce memory limits"""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                if RESOURCE_AVAILABLE and not IS_WINDOWS:
                    resource.setrlimit(resource.RLIMIT_AS, 
                        (self.max_memory, self.max_memory))
                elif IS_WINDOWS:
                    process = psutil.Process()
                    if process.memory_info().rss > self.max_memory:
                        self.logger.warning("Memory usage high on Windows")
                        return "Error: System memory limit approached"
                        
            except Exception as e:
                self.logger.warning(f"Memory limit setting failed: {e}")
            
            try:
                return func(self, *args, **kwargs)
            except MemoryError:
                self.logger.critical("Memory limit exceeded")
                return "Error: Calculation exceeded memory limits"
            finally:
                if 'result' in locals():
                    del locals()['result']
        return wrapper

    def parse_expression(self, expression: str) -> str:
        """Parse and prepare the expression for SymPy evaluation."""
        try:
            self.logger.debug(f"Parsing expression: {expression}")
            
            operators = ['>=', '<=', '>', '<', '=']
            if any(op in expression for op in operators):
                return self._handle_equation(expression)
            
            expression = ' '.join(expression.split())
            
            expression = self._parse_sum(expression)
            expression, is_harmonic = self._parse_harmonic(expression)
            if is_harmonic:
                return expression
            expression = self._parse_product(expression)
            
            if 'C' in expression:
                expression = self._parse_combination(expression)
            
            elif 'P' in expression:
                expression = self._parse_permutation(expression)
            
            elif expression.startswith('lim'):
                expression = self._parse_limit(expression)
            
            elif any(x in expression for x in ['d/dx', 'd/dy', 'diff(']):
                expression = self._parse_derivative(expression)
            
            elif expression.startswith('int'):
                expression = self._parse_integral(expression)
            
            expression = expression.replace('°', '*pi/180')
            expression = expression.replace('pi', 'sympy.pi')
            expression = expression.replace('^', '**')
            
            self.math_dict['Sum'] = sympy.Sum
            self.math_dict['oo'] = sympy.oo
            
            self.logger.debug(f"Parsed expression: {expression}")
            return expression
            
        except Exception as e:
            self.logger.error(f"Error parsing expression: {e}")
            raise

    def _parse_sum(self, expression: str) -> str:
        """Handle sum expressions."""
        sum_pattern = r'sum\s*\(\s*(\d+)\s+to\s+(\d+|inf|Inf)\s*\)\s*([^\n]+)'
        
        def replace_sum(match):
            start = match.group(1)
            end = match.group(2).lower()
            expr_part = match.group(3)
            
            if end == 'inf':
                return f"Sum({expr_part}, (x, {start}, oo))"
            elif end == '-inf':
                return f"Sum({expr_part}, (x, {start}, -oo))"
            else:
                return f"Sum({expr_part}, (x, {start}, {end}))"
        
        if re.search(sum_pattern, expression):
            return re.sub(sum_pattern, replace_sum, expression)
        return expression

    def _parse_harmonic(self, expression: str) -> tuple[str, bool]:
        """Handle harmonic series expressions."""
        harmonic_pattern = r'sum\s*\(\s*(\d+)\s*to\s*(?:inf|Inf|(\d+))\s*\)\s*1/([a-zA-Z])'
        
        def replace_harmonic(match):
            start = int(match.group(1))
            end = match.group(2)
            var = match.group(3)
            
            if end is None:
                return "∞"
            else:
                if start == 1:
                    return f"log({end}) + 0.57721566490153286060"
                else:
                    return f"log({end}) - log({start-1})"
        
        if re.search(harmonic_pattern, expression):
            return re.sub(harmonic_pattern, replace_harmonic, expression), True
        return expression, False

    def _parse_product(self, expression: str) -> str:
        """Handle product expressions."""
        prod_pattern = r'prod\s*\(\s*(\d+)\s*to\s*(?:inf|Inf|(\d+))\s*\)\s*([^\n]+)'
        
        def replace_prod(match):
            start = match.group(1)
            end = match.group(2)
            expr_part = match.group(3)
            
            if end is None:
                return f"Product({expr_part}, (x, {start}, oo))"
            else:
                return f"Product({expr_part}, (x, {start}, {end}))"
        
        return re.sub(prod_pattern, replace_prod, expression)

    def _parse_combination(self, expression: str) -> str:
        try:
            match = re.search(r'(\d+)C(\d+)', expression)
            if match:
                n, r = match.groups()
                return expression.replace(f"{n}C{r}", f"binomial({n}, {r})")
            return expression
        except Exception as e:
            self.logger.error(f"Error parsing combination: {e}")
            raise ValueError(f"Invalid combination format: {expression}")

    def _parse_permutation(self, expression: str) -> str:
        try:
            match = re.search(r'(\d+)P(\d+)', expression)
            if match:
                n, r = match.groups()
                return expression.replace(f"{n}P{r}", f"factorial({n})/factorial({n}-{r})")
            return expression
        except Exception as e:
            self.logger.error(f"Error parsing permutation: {e}")
            raise ValueError(f"Invalid permutation format: {expression}")

    def _parse_integral(self, expression: str) -> str:
        try:
            expression = expression.strip()
            self.logger.debug(f"Processing integral expression: {expression}")
            
            definite_pattern = r'int\s*\(\s*([^,]+)\s+to\s+([^,)]+)\s*\)\s*([^\n]+?)\s*dx'
            definite_match = re.match(definite_pattern, expression)
            
            indefinite_pattern = r'int\s*([^\n]+?)\s*dx'
            indefinite_match = re.match(indefinite_pattern, expression)
            
            if definite_match:
                lower_limit = definite_match.group(1).strip()
                upper_limit = definite_match.group(2).strip()
                integrand = definite_match.group(3).strip()
                
                if not integrand:
                    raise ValueError("Missing integrand in definite integral")
                    
                integrand = self._fix_trig_expr(integrand)
                    
                if 'ln(' in integrand:
                    integrand = integrand.replace('ln(', 'log(')
                    
                self.logger.debug(f"Definite integral: from {lower_limit} to {upper_limit} of {integrand}")
                return f"Integral({integrand}, (x, {lower_limit}, {upper_limit}))"
                
            elif indefinite_match:
                integrand = indefinite_match.group(1).strip()
                self.logger.debug(f"Found indefinite integral with integrand: {integrand}")
                
                if not integrand:
                    raise ValueError("Missing integrand in indefinite integral")
                    
                integrand = self._fix_trig_expr(integrand)
                    
                if 'ln(' in integrand:
                    integrand = integrand.replace('ln(', 'log(')
                    
                self.logger.debug(f"Indefinite integral of {integrand}")
                return f"Integral({integrand}, x)"
                
            else:
                raise ValueError("Invalid integral format")
                
        except Exception as e:
            self.logger.error(f"Error parsing integral: {e}")
            raise ValueError(f"Invalid integral format: {str(e)}")

    def _fix_trig_expr(self, expr: str) -> str:
        """Fix trigonometric expressions by adding multiplication operator."""
        expr = re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', expr)
        
        trig_funcs = ['sin', 'cos', 'tan', 'csc', 'sec', 'cot']
        for func in trig_funcs:
            expr = re.sub(rf'(\d+)({func})', rf'\1*{func}', expr)
        
        for func in trig_funcs:
            expr = re.sub(rf'{func}\((\d+)([a-zA-Z])\)', rf'{func}(\1*\2)', expr)
        
        return expr

    def _parse_derivative(self, expression: str) -> str:
        """Parse derivative expressions."""
        try:
            if 'd/dx' in expression:
                match = re.search(r'd/dx\((.*?)\)', expression) or re.search(r'd/dx\s*(.*)', expression)
                if match:
                    func = match.group(1).strip()
                    order = expression.count('d/dx')
                    return f"diff({func}, x, {order})" if order > 1 else f"diff({func}, x)"
                
            elif 'd/dy' in expression:
                match = re.search(r'd/dy\((.*?)\)', expression) or re.search(r'd/dy\s*(.*)', expression)
                if match:
                    func = match.group(1).strip()
                    order = expression.count('d/dy')
                    return f"diff({func}, y, {order})" if order > 1 else f"diff({func}, y)"
                    
            elif expression.startswith('diff('):
                return expression
                
            self.logger.debug(f"No valid derivative format found in: {expression}")
            return expression
            
        except Exception as e:
            self.logger.error(f"Error parsing derivative: {e}")
            raise ValueError(f"Error parsing derivative: {e}")

    def _parse_limit(self, expression: str, angle_mode: str = 'Radian') -> str:
        """Parse limit expressions."""
        try:
            limit_pattern = r'(?i)lim\s*\(\s*([a-zA-Z])\s*to\s*' + \
                           r'(' + \
                           r'[-+]?\d*\.?\d+|' + \
                           r'[-+]?(?:inf(?:ty|inity)?)|' + \
                           r'[a-zA-Z][a-zA-Z0-9]*|' + \
                           r'(?:sin|cos|tan|csc|sec|cot|arcsin|arccos|arctan|arcsec|arccsc|arccot|ln|log|log\d+|sqrt|exp)\s*\([^)]+\)|' + \
                           r'e\^?[^,\s)]*' + \
                           r')([+-])?\s*\)\s*' + \
                           r'((?:[^()]+|\((?:[^()]+|\([^()]*\))*\))*)'
            
            match = re.match(limit_pattern, expression)
            
            if match:
                var, approach, side, function = match.groups()
                
                # Handle infinity in approach value
                if isinstance(approach, str) and re.match(r'(?i)inf(?:ty|inity)?', approach):
                    approach = 'oo'
                elif isinstance(approach, str) and re.match(r'(?i)[+-]inf(?:ty|inity)?', approach):
                    sign = approach[0]
                    approach = f'{sign}oo'
                
                # Handle trigonometric functions in approach value
                if angle_mode == 'Degree' and '(' in approach:
                    # Convert regular trig functions to use radians
                    if any(trig in approach for trig in ['sin(', 'cos(', 'tan(', 'sec(', 'csc(', 'cot(']):
                        approach = re.sub(r'sin\((.*?)\)', r'sin(pi/180*(\1))', approach)
                        approach = re.sub(r'cos\((.*?)\)', r'cos(pi/180*(\1))', approach)
                        approach = re.sub(r'tan\((.*?)\)', r'tan(pi/180*(\1))', approach)
                        approach = re.sub(r'sec\((.*?)\)', r'sec(pi/180*(\1))', approach)
                        approach = re.sub(r'csc\((.*?)\)', r'csc(pi/180*(\1))', approach)
                        approach = re.sub(r'cot\((.*?)\)', r'cot(pi/180*(\1))', approach)
                    
                    # Evaluate the approach if it contains trigonometric functions
                    if any(trig in approach for trig in ['sin(', 'cos(', 'tan(', 'sec(', 'csc(', 'cot(']):
                        approach_expr = sympify(approach, locals=self.math_dict)
                        approach = str(approach_expr.evalf())
                
                # Format the function part
                function = function.strip()
                
                # Convert function part to radians if in degree mode
                if angle_mode == 'Degree':
                    if any(trig in function for trig in ['sin(', 'cos(', 'tan(', 'sec(', 'csc(', 'cot(']):
                        function = re.sub(r'sin\((.*?)\)', r'sin(pi/180*(\1))', function)
                        function = re.sub(r'cos\((.*?)\)', r'cos(pi/180*(\1))', function)
                        function = re.sub(r'tan\((.*?)\)', r'tan(pi/180*(\1))', function)
                        function = re.sub(r'sec\((.*?)\)', r'sec(pi/180*(\1))', function)
                        function = re.sub(r'csc\((.*?)\)', r'csc(pi/180*(\1))', function)
                        function = re.sub(r'cot\((.*?)\)', r'cot(pi/180*(\1))', function)
                
                self.logger.debug(f"Limit: {var} -> {approach} of {function}")
                
                # Create limit expression
                limit_expr = f"limit({function}, {var}, {approach})"
                if side:
                    if side == '+':
                        limit_expr = f"limit({function}, {var}, {approach}, dir='+')"
                    elif side == '-':
                        limit_expr = f"limit({function}, {var}, {approach}, dir='-')"
                
                self.logger.debug(f"Limit expression: {limit_expr}")
                result = sympify(limit_expr, locals=self.math_dict)
                
                # Handle infinity in result
                if result == sympy.oo:
                    return "∞"
                elif result == -sympy.oo:
                    return "-∞"
                
                if isinstance(result, sympy.Basic):
                    numeric_result = float(N(result))
                    if angle_mode == 'Degree' and any(trig in function for trig in ['asin', 'acos', 'atan', 'asec', 'acsc', 'acot']):
                        numeric_result = float(numeric_result * 180 / float(N(pi)))
                    
                    # Format based on magnitude
                    if abs(numeric_result) > 1e10 or (0 < abs(numeric_result) < 1e-10):
                        formatted = f"{numeric_result:.8e}"
                        parts = formatted.split('e')
                        if len(parts) == 2:
                            return f"{parts[0]} e{parts[1]}"
                        else:
                            return formatted
                    else:
                        return f"{numeric_result:8f}".rstrip('0').rstrip('.')
                
                return str(result)
                
            else:
                raise ValueError("Invalid limit format")
                
        except Exception as e:
            self.logger.error(f"Error parsing limit: {e}")
            raise ValueError(f"Invalid limit format: {str(e)}")

    def _check_memory(self) -> bool:
        """Check if memory usage is within limits."""
        try:
            if self._process:
                return self._process.memory_info().rss <= self.max_memory
            return True
        except Exception as e:
            self.logger.warning(f"Memory check failed: {e}")
            return True

    def evaluate(self, expression: str, angle_mode: str = 'Radian') -> str:
        """Evaluate the SymPy expression and return the result."""
        try:
            # Add memory check at start
            if not self._check_memory():
                return "Error: System memory limit approached"

            if not expression or expression.isspace():
                raise ValueError("Empty expression provided")
            
            self.logger.debug(f"Evaluating expression: {expression}")
            
            # Check if this is a limit expression
            if expression.startswith('lim'):
                result = self._parse_limit(expression, angle_mode)
                return result
            
            if not re.match(r'^[0-9a-zA-Z\s\+\-\*\/\^\(\)\.=<>!]+$', expression):
                raise ValueError("Invalid characters in expression")
            
            parsed_expr = self.parse_expression(expression)
            
            if not parsed_expr or parsed_expr.isspace():
                raise ValueError("Failed to parse expression")
            
            result = sympify(parsed_expr, locals=self.math_dict)
            
            # Add memory check after major computations
            if isinstance(result, (sympy.Integral, sympy.Derivative)):
                if not self._check_memory():
                    return "Error: Insufficient memory for computation"
                result = result.doit()
            
            try:
                if result == sympy.oo or result == float('inf'):
                    return "∞"
                elif result == -sympy.oo or result == float('-inf'):
                    return "-∞"
                elif result.is_number:
                    numeric_result = float(N(result))
                    if angle_mode == 'Degree' and any(trig in expression for trig in ['asin', 'acos', 'atan']):
                        numeric_result = float(numeric_result * 180 / float(N(pi)))
                    
                    # Format based on magnitude
                    if abs(numeric_result) > 1e10 or (0 < abs(numeric_result) < 1e-10):
                        formatted = f"{numeric_result:.8e}"
                        parts = formatted.split('e')
                        if len(parts) == 2:
                            result = f"{parts[0]} e{parts[1]}"
                        else:
                            result = formatted
                    else:
                        result = f"{numeric_result:.8f}".rstrip('0').rstrip('.')
                else:
                    # For symbolic results, try to simplify
                    result = sympy.simplify(result)
                    result = str(result)
                
                # Add output length check
                if len(result) > self.max_output_length:
                    self.logger.warning("Truncating large output")
                    result = result[:self.max_output_length] + "... [truncated]"
                
                result = self._clean_display(result, angle_mode)
                
                # Final memory check
                if not self._check_memory():
                    self.logger.warning("Memory limit exceeded after computation")
                
                self.logger.debug(f"Evaluation result: {result}")
                return result
                
            except Exception as e:
                self.logger.error(f"Error evaluating expression: {e}")
                return f"Error: {str(e)}"
            
        except ValueError as ve:
            self.logger.error(f"Invalid input: {ve}")
            return f"Error: {str(ve)}"
        except Exception as e:
            self.logger.error(f"Error evaluating expression: {e}")
            return f"Error: {str(e)}"

    def _clean_display(self, result: str, angle_mode: str) -> str:
        """Clean up the display format of the result."""
        try:
            # Apply replacements in sequence
            result = result.replace('oo', '∞')
            result = re.sub(r'sqrt\((\d+)\)', r'√\1', result)
            result = result.replace('**', '^')
            result = result.replace('exp(1)', 'e')
            result = result.replace('log(', 'ln(')
            
            result = re.sub(
                r'(?<!\()(-?\d*\.?\d+\s*[+\-]\s*[√]?\d*\.?\d+)(?![\d\s)])', 
                r'(\1)', 
                result
            )
            
            # Handle trigonometric functions in degree mode
            if angle_mode == 'Degree':
                result = re.sub(r'(sin|cos|tan)\((.*?)\)', r'\1_{deg}(\2)', result)
            
            # Final cleanup of redundant parentheses
            result = re.sub(r'\((√\d+)\)', r'\1', result)
            result = re.sub(r'\((\d+)\)', r'\1', result)
            
            # Unify logical operators: replace SymPy's '|' and '&' with "or" and "and" respectively.
            result = result.replace('|', ' or ')
            result = result.replace('&', ' and ')
            
            # Remove any extra whitespace.
            result = re.sub(r'\s+', ' ', result).strip()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error cleaning display: {e}")
            return result

    def _handle_equation(self, expression: str) -> str:
        """
        Handle equation and inequality expressions for SymPy evaluation.
        
        Args:
            expression (str): Input equation/inequality expression
            
        Returns:
            str: Processed equation for SymPy
        """
        try:
            from sympy import Symbol, Eq, solve, sympify
             
            sympy_eq_symbols = {
                r'\geq': '>=',
                r'\leq': '<=',
                r'\neq': '!=',
                r'\gg': '>',
                r'\ll': '<',
                r'\approx': '==',
                r'\equiv': '==',
                r'\propto': '==',
                r'\sim': '=='
            }
            
            result = expression
            for latex_sym, sympy_sym in sympy_eq_symbols.items():
                result = result.replace(latex_sym, sympy_sym)
            
            result = result.replace('^', '**')
            result = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', result)
            
            operators = ['>=', '<=', '>', '<', '=']
            for op in operators:
                if op in result:
                    left_side = result[:result.find(op)].strip()
                    right_side = result[result.find(op) + len(op):].strip()
                    
                    if any(trig in left_side for trig in ['sin', 'cos', 'tan']):
                        equation = Eq(sympify(left_side, locals=self.math_dict),
                                      sympify(right_side, locals=self.math_dict))
                        variables = [str(s) for s in equation.free_symbols]
                        if variables:
                            var = Symbol(variables[0])
                            solution = solve(equation, var)
                            if solution:
                                if op == '=':
                                    solutions = [f"{var}{i+1} = {sol}" for i, sol in enumerate(solution)]
                                    return ', '.join(solutions)
                                else:
                                    return ', '.join([f"{sol}" for sol in solution])
                            return f"No solution found for {equation}"
                    
                    if not right_side or right_side == '0':
                        equation = left_side
                    else:
                        equation = f"{left_side}-({right_side})"
                        equation = equation.replace('-(0)', '')
                    
                    variables = list(Symbol(s) for s in re.findall(r'[a-zA-Z]', equation))
                    if variables:
                        var_symbol = variables[0]
                        if op in ['>', '<', '>=', '<=']:
                            expr = sympify(equation)
                            
                            # Solve the inequality directly.
                            if op == '>':
                                solution = solve(expr > 0, var_symbol)
                            elif op == '<':
                                solution = solve(expr < 0, var_symbol)
                            elif op == '>=':
                                solution = solve(expr >= 0, var_symbol)
                            elif op == '<=':
                                solution = solve(expr <= 0, var_symbol)
                            
                            if isinstance(solution, list):
                                solution.sort(key=lambda x: float(x.evalf()))
                                ranges = []
                                for i in range(len(solution)):
                                    bound = str(solution[i])
                                    
                                    if i == 0:
                                        if op in ['<', '<=']:
                                            ranges.append(f"x < {bound}")
                                        else:
                                            ranges.append(f"x > {bound}")
                                    elif i == len(solution) - 1:
                                        if op in ['<', '<=']:
                                            ranges.append(f"x > {bound}")
                                        else:
                                            ranges.append(f"x < {bound}")
                                return " or ".join(ranges)
                            return str(solution)
                        else:
                            return f"solve({equation}, {var_symbol})"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error solving equation: {e}")
            return f"Error: {str(e)}"
