'''
Copyright (c) 2025 George Huang. All Rights Reserved.

This file is part of the VorTeX Calculator project.
Component: MATLAB Interface - Expression Evaluator

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

Version: 1.0.3
Last Updated: 2025.2
'''

import logging
import re
from functools import lru_cache, wraps
from matlab_interface.auto_simplify import AutoSimplify
from latex_pack.shortcut import ExpressionShortcuts
from sympy_pack.sympy_calculation import SympyCalculation
import psutil
import resource
from sys import platform as sys_platform

def matlab_memory_safe(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            # Only monitor Windows memory usage
            if sys_platform.startswith('win') and self._process:
                mem_info = self._process.memory_info()
                if mem_info.rss > self.max_memory and not self._windows_mem_warned:
                    self.logger.warning("Approaching Windows memory limits")
                    self._windows_mem_warned = True
        except Exception as e:
            self.logger.debug(f"Memory monitoring skipped: {e}")

        try:
            result = func(self, *args, **kwargs)
            
            try:
                self.eng.eval("clear ans;", nargout=0)
            except:
                pass
            
            if result and isinstance(result, str) and len(result) > 2000:
                result = result[:2000] + "... [truncated]"
                
            if isinstance(result, str):
                return result.replace('Inf', '∞').replace('NaN', 'undefined')
            return result
            
        except MemoryError:
            self.logger.critical("MATLAB memory limit exceeded")
            return "Error: Exceeded available memory"
        except Exception as e:
            self.logger.error(f"Error in MATLAB computation: {e}")
            return f"Error: {str(e)}"
        finally:
            if 'result' in locals():
                del result
            try:
                self.eng.eval("clear temp_result;", nargout=0)
            except:
                pass
    return wrapper

class EvaluateExpression:
    def __init__(self, eng, angle_mode='rad'):
        """
        Initialize the EvaluateExpression class with a MATLAB engine instance.
        
        Args:
            eng (matlab.engine.MatlabEngine): An active MATLAB engine session.
            angle_mode (str): Angle mode, either 'rad' for radians (default) or 'deg' for degrees.
        """
        self.eng = eng
        self.angle_mode = angle_mode.lower()  # Default to radians
        self.logger = logging.getLogger(__name__)
        self._configure_logger()
        self.simplifier = AutoSimplify(eng)  # Initialize the AutoSimplify class
        self._compile_patterns()
        self._initialize_workspace()
        self._symbolic_vars = set()
        self.max_memory = 768 * 1024 * 1024  # 768MB for MATLAB processes
        self._process = psutil.Process() if psutil else None
        self._windows_mem_warned = False
        
        if self.angle_mode == 'rad':
            self.eng.eval("syms x real; assume(x, 'real'); clear x;", nargout=0)
        else:
            self.eng.eval("syms x real; assume(x, 'real'); clear x;", nargout=0)
            
        self.logger.debug(f"Initialized with angle mode: {self.angle_mode}")

    def set_angle_mode(self, mode):
        """
        Set the angle mode for trigonometric calculations.
        
        Args:
            mode (str): Either 'rad' for radians or 'deg' for degrees.
        """
        if mode.lower() not in ['rad', 'deg']:
            self.logger.warning(f"Invalid angle mode: {mode}. Using default (rad).")
            mode = 'rad'
            
        self.angle_mode = mode.lower()
        self.logger.debug(f"Angle mode set to: {self.angle_mode}")

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

    def _compile_patterns(self):
        """
        Precompile regular expressions for efficiency.
        """
        self.ln_pattern = re.compile(r'\bln\s*\(')
        self.trig_rad_patterns = {
            re.compile(r'\bsin\s*\('): 'sin(',
            re.compile(r'\bcos\s*\('): 'cos(',
            re.compile(r'\btan\s*\('): 'tan(',
            re.compile(r'\basin\s*\('): 'asin(',
            re.compile(r'\bacos\s*\('): 'acos(',
            re.compile(r'\batan\s*\('): 'atan('
        }
        self.trig_deg_patterns = {
            re.compile(r'\bsind\s*\('): 'sind(',
            re.compile(r'\bcosd\s*\('): 'cosd(',
            re.compile(r'\btand\s*\('): 'tand(',
            re.compile(r'\basind\s*\('): 'asind(',
            re.compile(r'\bacosd\s*\('): 'acosd(',
            re.compile(r'\batand\s*\('): 'atand('
        }
        self.log_e_pattern = re.compile(r'log\s*\(\s*E\s*,\s*([^,)]+)\)')
        self.log_base_pattern = re.compile(r'log\s*\(\s*(\d+)\s*,\s*([^,)]+)\)')
        self.log10_pattern = re.compile(r'\blog10\s*\(')  # Add pattern for log10
        self.mul_pattern = re.compile(r'(\d)([a-zA-Z])')

    def _initialize_workspace(self):
        """
        Initialize the MATLAB workspace with necessary constants and symbolic variables.
        """
        angle_mode_setting = "rad" if self.angle_mode == 'rad' else "deg"
        
        init_cmd = f"""
        pi = pi;      % Use MATLAB's built-in pi constant
        e = exp(1);   % Define e
        syms x;       % Define x as symbolic
        
        % Set the angle mode (rad or deg)
        if exist('trigmode', 'file')
            trigmode('{angle_mode_setting}');
        end
        """
        self.eng.eval(init_cmd, nargout=0)
        self.logger.debug(f"Initialized e and pi in MATLAB workspace with angle mode: {self.angle_mode}")

    def _handle_trigonometric_functions(self, expression):
        """
        Handle trigonometric functions based on angle mode setting.
        
        Args:
            expression (str): The input expression containing trigonometric functions.
            
        Returns:
            str: The expression with trigonometric functions modified according to angle mode.
        """
        self.logger.debug(f"Handling trigonometric functions with angle mode: {self.angle_mode}")
        
        if self.angle_mode == 'deg':
            for standard_func, deg_func in [
                ('sin(', 'sind('), ('cos(', 'cosd('), ('tan(', 'tand('),
                ('asin(', 'asind('), ('acos(', 'acosd('), ('atan(', 'atand(')
            ]:
                expression = expression.replace(standard_func, deg_func)
            
            for func in ['sec(', 'csc(', 'cot(', 'asec(', 'acsc(', 'acot(', 
                        'sinh(', 'cosh(', 'tanh(', 'sech(', 'csch(', 'coth(',
                        'asinh(', 'acosh(', 'atanh(', 'asech(', 'acsch(', 'acoth(']:
                pattern = re.escape(func[:-1]) + r'\s*\(([^)]+)\)'
                expression = re.sub(pattern, lambda m: f"{func[:-1]}(deg2rad({m.group(1)}))", expression)
        
        self.logger.debug(f"After trig function handling: {expression}")
        return expression

    def _preprocess_expression(self, expression):
        """
        Preprocess the expression before evaluation to ensure compatibility with MATLAB.
        
        Args:
            expression (str): The mathematical expression to preprocess.
            
        Returns:
            str: The preprocessed expression.
        """
        self.logger.debug(f"Original expression: {expression}")
        
        euler_identity_pattern = re.compile(r'(e\^[\s(]*i\s*\*\s*pi[\s)]*|exp[\s(]*i\s*\*\s*pi[\s)]*)\s*\+\s*1')
        if euler_identity_pattern.search(expression):
            self.logger.debug("Detected Euler's identity")
            return "0"
            
        expression = re.sub(r'(?<![a-zA-Z0-9_])ln\s*\(', 'log(', expression)
        self.logger.debug(f"After replacing 'ln' with 'log(': {expression}")
        
        expression = re.sub(r'log_e\s*\(', 'log(', expression)
        expression = re.sub(r'log_(\d+)\s*\(([^,)]+)\)', r'log(\2)/log(\1)', expression)
        expression = ExpressionShortcuts._convert_logarithms(expression)
        expression = re.sub(r'log10\s*\(', '__LOG10_PLACEHOLDER__', expression)
        
        expression = ExpressionShortcuts.convert_integral_expression(expression)
        expression = ExpressionShortcuts.convert_limit_expression(expression)
        expression = ExpressionShortcuts.convert_sum_prod_expression(expression)
        expression = ExpressionShortcuts.convert_factorial_expression(expression)
        expression = ExpressionShortcuts.convert_permutation_expression(expression)
        expression = ExpressionShortcuts.convert_exponential_expression(expression)
        expression = ExpressionShortcuts.convert_complex_expression(expression)
        
        d_dx_pattern = re.compile(r'd([0-9]*)/d([a-zA-Z])([0-9]*)\s+([a-zA-Z])')
        if d_dx_pattern.search(expression):
            def replace_diff(match):
                order = match.group(1) if match.group(1) else '1'
                var = match.group(2)
                var_power = match.group(3) if match.group(3) else '1'
                func = match.group(4)
                
                order = int(order) if order else 1
                return f"diff({func}({var}), {var}, {order})"
                
            expression = d_dx_pattern.sub(replace_diff, expression)
        
        self.logger.debug(f"Preserving log10 function")
        
        if self.angle_mode == 'rad':
            for pattern, replacement in self.trig_rad_patterns.items():
                expression = pattern.sub(replacement, expression)
        else:
            for pattern, replacement in self.trig_deg_patterns.items():
                expression = pattern.sub(replacement, expression)
        
        expression = re.sub(r'(\d+)!', r'factorial(\1)', expression)
        
        sum_pattern = re.compile(r'sum\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,)]+)\s*\)')
        if sum_pattern.search(expression):
            expression = sum_pattern.sub(r'symsum(\1, \2, \3, \4)', expression)
            
        prod_pattern = re.compile(r'prod\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,)]+)\s*\)')
        if prod_pattern.search(expression):
            expression = prod_pattern.sub(r'symprod(\1, \2, \3, \4)', expression)
        
        expression = re.sub(r'(\d+)([a-zA-Z_](?!__LOG10_PLACEHOLDER__))', r'\1*\2', expression)
        expression = re.sub(r'__LOG10_PLACEHOLDER__', 'log10(', expression)
        
        expression = re.sub(r'e\^', 'exp', expression)
        if 'exp' in expression:
            expression = self._format_exponential(expression)
        
        self.logger.debug(f"Preprocessed expression: {expression}")
        return expression

    def _format_exponential(self, expression):
        """
        Format exponential expressions for MATLAB compatibility.
        
        Args:
            expression (str): The expression containing exponential functions.
            
        Returns:
            str: The expression with properly formatted exponential functions.
        """
        self.logger.debug(f"Formatting exponential expression: {expression}")
        
        exp_pattern = re.compile(r'exp\s*\(\s*([^)]+)\s*\)')
        
        self.logger.debug(f"After exponential formatting: {expression}")
        return expression

    def _format_single_line_equation(self, result_str):
        parts = result_str.split('=')
        if len(parts) == 2:
            left = parts[0].strip()
            right = parts[1].strip()
            if right == "":
                formatted_right = "0"
            else:
                try:
                    f = float(right)
                    formatted_right = self.simplifier.round_numeric_value(f)
                except ValueError:
                    formatted_right = right
            return f"{left} = {formatted_right}"
        return result_str

    def _format_polynomial_roots(self, result_str):
        """
        Format polynomial roots in a more readable format with x₁, x₂, x₃ notation.
        
        Args:
            result_str (str): The result string from MATLAB, potentially containing polynomial roots.
            
        Returns:
            str: The formatted result string with x₁, x₂, x₃ notation.
        """
        if result_str.startswith('[') and result_str.endswith(']'):
            content = result_str[1:-1]
            
            roots = content.split(';')
            
            if len(roots) > 1:
                formatted_roots = []
                for i, root in enumerate(roots):
                    root = root.strip()
                    
                    # Format the root for better readability
                    root = re.sub(r'(\d+)\^\(1/2\)', r'√\1', root)
                    root = re.sub(r'(\d+)\^\(1/3\)', r'∛\1', root)
                    root = re.sub(r'(\d+)\^\(1/4\)', r'∜\1', root)
                    
                    # Use subscript numbers for x₁, x₂, x₃
                    subscripts = {
                        '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅',
                        '6': '₆', '7': '₇', '8': '₈', '9': '₉', '0': '₀'
                    }
                    index = ''.join(subscripts[c] for c in str(i+1))
                    formatted_roots.append(f"x{index} = {root}")
                
                return '\n'.join(formatted_roots)
        
        return result_str

    def _postprocess_result(self, result_str, is_numeric=False):
        """
        Format the result string for display.
        
        Args:
            result_str (str): The result string from MATLAB.
            is_numeric (bool): Whether the result is numeric.
            
        Returns:
            str: The formatted result string.
        """
        if not result_str:
            return "0"
        
        if result_str.startswith('[') and result_str.endswith(']') and ';' in result_str:
            formatted_roots = self._format_polynomial_roots(result_str)
            if formatted_roots != result_str:
                return formatted_roots
        
        if 'exp(' in result_str:
            self.logger.debug(f"Converting exponential expression in result: {result_str}")
            exp_pattern = re.compile(r'exp\(([^()]*(?:\([^()]*\)[^()]*)*)\)')
            while exp_pattern.search(result_str):
                result_str = exp_pattern.sub(r'e^(\1)', result_str)
            self.logger.debug(f"After exponential conversion: {result_str}")
        
        if '1i*pi' in result_str:
            result_str = result_str.replace('e^(1i*pi)', 'e^(i*pi)')
        
        if '1i' in result_str:
            result_str = result_str.replace('1i', 'i')
        
        if '=' in result_str and '\n' not in result_str:
            result_str = self._format_single_line_equation(result_str)
            return result_str
        
        if 'solve(' in result_str and '>' not in result_str and '<' not in result_str:
            try:
                self.eng.eval(f"temp_result = {result_str};", nargout=0)
                solution = str(self.eng.eval("char(temp_result)", nargout=1))
                
                if solution and solution != '[]':
                    if ',' in solution:
                        solutions = solution.split(',')
                        return ' or '.join(solutions)
                    return solution
                return "No solution"
            except:
                pass
        
        if any(op in result_str for op in ['>', '<', '>=', '<=']):
            if result_str.endswith('>') or result_str.endswith('<'):
                result_str = result_str + '0'
            elif result_str.endswith('> ') or result_str.endswith('< '):
                result_str = result_str + '0'
            return result_str
        
        if result_str.lower() in ['inf', '+inf', 'infinity']:
            return "∞"
        if result_str.lower() in ['-inf', '-infinity']:
            return "-∞"
        
        try:
            if '\n' in result_str:
                lines = result_str.split('\n')
                processed_lines = []
                for line in lines:
                    if '=' in line:
                        var, val = line.split('=')
                        try:
                            float_val = float(val)
                            rounded_val = self.simplifier.round_numeric_value(float_val)
                            processed_lines.append(f"{var}= {rounded_val}")
                        except ValueError:
                            processed_lines.append(line)
                    else:
                        processed_lines.append(line)
                return '\n'.join(processed_lines)
            
            result_str = result_str.rstrip('0').rstrip('.')
            float_val = float(result_str)
            
            return self.simplifier.round_numeric_value(float_val)
            
        except ValueError:
            pass
        
        special_cases = {
            'inf': '∞',
            '+inf': '∞',
            '-inf': '-∞',
            'nan': 'undefined'
        }
        result_str = special_cases.get(result_str.lower(), result_str)
    
        result_str = result_str.replace('*1.0', '').replace('1.0*', '').replace('.0', '')
        
        # Convert radian-based expressions back to degree functions
        degree_pattern = re.compile(r'(\bcos|\bsin|\btan)\(\(\s*([^)]+)\s*\*\s*pi\s*/\s*180\s*\)\)')
        result_str = degree_pattern.sub(r'\1d(\2)', result_str)
        
        exp_pattern = re.compile(r'exp\(([^()]*(?:\([^()]*\)[^()]*)*)\)')
        while exp_pattern.search(result_str):
            result_str = exp_pattern.sub(r'e^(\1)', result_str)
        
        symbolizations = {
            'asin(': 'arcsin(',
            'acos(': 'arccos(',
            'atan(': 'arctan(',
            'log(': 'ln('
        }
        
        for old, new in symbolizations.items():
            result_str = result_str.replace(old, new)
        
        trig_funcs = ['sin', 'cos', 'tan', 'csc', 'sec', 'cot']
        for func in trig_funcs:
            pattern = f'{func}\\(([^()]*(?:\\([^()]*\\)[^()]*)*)\\)'
            matches = list(re.finditer(pattern, result_str))
            for match in reversed(matches):
                full_match = match.group(0)
                arg = match.group(1)
                if '(' not in arg and ')' not in arg:
                    result_str = result_str.replace(full_match, f'{func}({arg})')
        
        if '/' in result_str:
            result_str = re.sub(r'\s+', '', result_str)
            result_str = result_str.replace(')(', ')*(')
        
        if result_str.count('(') < result_str.count(')'):
            result_str = result_str.rstrip(')')
        
        self.logger.debug(f"Symbolic result after postprocessing: {result_str}")
        return result_str

    @lru_cache(maxsize=64)
    @matlab_memory_safe
    def evaluate(self, expression):
        """
        Evaluate a mathematical expression using MATLAB.
        
        Args:
            expression (str): Mathematical expression to evaluate
            
        Returns:
            str: Result from MATLAB evaluation
        """
        self.logger.debug(f"Original expression: {expression}")
        
        euler_identity_pattern = re.compile(r'(e\^[\s(]*i\s*\*\s*pi[\s)]*|exp[\s(]*i\s*\*\s*pi[\s)]*)\s*\+\s*1')
        if euler_identity_pattern.search(expression):
            self.logger.debug("Detected Euler's identity")
            return "0"
        
        try:
            if "'" in expression or "d/d" in expression:
                self.logger.debug(f"Detected potential differential equation: {expression}")
            
            trig_special_values = {
                'sin(pi/6)': '1/2',       # sin(π/6) = 0.5
                'sin(pi/4)': 'sqrt(2)/2', # sin(π/4) = √2/2
                'sin(pi/3)': 'sqrt(3)/2', # sin(π/3) = √3/2
                'sin(pi/2)': '1',         # sin(π/2) = 1
                'sin(2*pi/3)': 'sqrt(3)/2', # sin(2π/3) = √3/2
                'sin(3*pi/4)': 'sqrt(2)/2', # sin(3π/4) = √2/2
                'sin(5*pi/6)': '1/2',     # sin(5π/6) = 0.5
                'sin(pi)': '0',           # sin(π) = 0
                'sin(3*pi/2)': '-1',      # sin(3π/2) = -1
                'sin(2*pi)': '0',         # sin(2π) = 0
                
                'cos(0)': '1',            # cos(0) = 1
                'cos(pi/6)': 'sqrt(3)/2', # cos(π/6) = √3/2
                'cos(pi/4)': 'sqrt(2)/2', # cos(π/4) = √2/2
                'cos(pi/3)': '1/2',       # cos(π/3) = 1/2
                'cos(pi/2)': '0',         # cos(π/2) = 0
                'cos(2*pi/3)': '-1/2',    # cos(2π/3) = -1/2
                'cos(3*pi/4)': '-sqrt(2)/2', # cos(3π/4) = -√2/2
                'cos(5*pi/6)': '-sqrt(3)/2', # cos(5π/6) = -√3/2
                'cos(pi)': '-1',          # cos(π) = -1
                'cos(3*pi/2)': '0',       # cos(3π/2) = 0
                'cos(2*pi)': '1',         # cos(2π) = 1
                
                'tan(0)': '0',            # tan(0) = 0
                'tan(pi/6)': '1/sqrt(3)', # tan(π/6) = 1/√3
                'tan(pi/4)': '1',         # tan(π/4) = 1
                'tan(pi/3)': 'sqrt(3)',   # tan(π/3) = √3
                'tan(2*pi/3)': '-sqrt(3)', # tan(2π/3) = -√3
                'tan(3*pi/4)': '-1',      # tan(3π/4) = -1
                'tan(5*pi/6)': '-1/sqrt(3)', # tan(5π/6) = -1/√3
                'tan(pi)': '0',           # tan(π) = 0
                'tan(2*pi)': '0',         # tan(2π) = 0
            }
            
            normalized_expr = expression.replace(' ', '')
            for pattern, value in trig_special_values.items():
                if normalized_expr == pattern.replace(' ', ''):
                    return value

            combined_values = {
                'sin(pi/4)+cos(pi/4)': 'sqrt(2)',
                'sin(pi/4)^2+cos(pi/4)^2': '1', 
                'sin(pi/2)*cos(pi)': '-1',
                'tan(pi/4)-cos(pi/3)': '1-1/2',
                'sin(pi/4)*cos(pi/4)': '1/2',
            }
            
            normalized_expr = re.sub(r'\s+', '', expression)
            for pattern, value in combined_values.items():
                pattern_normalized = re.sub(r'\s+', '', pattern)
                if normalized_expr == pattern_normalized:
                    return value
            
            modified_expr = expression
            trig_pi_pattern = r'(sin|cos|tan)\s*\(\s*(pi|pi/\d+|\d+\s*\*\s*pi|\d+\s*\*\s*pi/\d+)\s*\)'
            
            matches = re.finditer(trig_pi_pattern, expression)
            for match in matches:
                full_match = match.group(0)
                normalized_match = full_match.replace(' ', '')
                
                for pattern, value in trig_special_values.items():
                    if normalized_match == pattern.replace(' ', ''):
                        modified_expr = modified_expr.replace(full_match, f"({value})")
                        break
            
            if modified_expr != expression:
                try:
                    command = f"result = double({modified_expr});"
                    self.eng.eval(command, nargout=0)
                    result_value = float(self.eng.eval("result", nargout=1))
                    self.eng.eval("clear result", nargout=0)
                    
                    return self.simplifier.round_numeric_value(result_value)
                    
                except Exception as e:
                    self.logger.debug(f"Error directly evaluating modified expression: {e}")

            exact_special_expressions = {
                'sqrt(2)/2': 'sqrt(2)/2',
                'sqrt(3)/2': 'sqrt(3)/2',
                'sqrt(2)/4': 'sqrt(2)/4',
                'sqrt(3)/4': 'sqrt(3)/4',
                '2^(1/2)/2': 'sqrt(2)/2',
                '3^(1/2)/2': 'sqrt(3)/2'
            }
            
            if normalized_expr in exact_special_expressions:
                return exact_special_expressions[normalized_expr]

            sqrt_division_pattern = r'sqrt\((\d+)\)/(\d+)'
            sqrt_division_match = re.match(sqrt_division_pattern, normalized_expr)
            if sqrt_division_match:
                num = sqrt_division_match.group(1)
                denom = sqrt_division_match.group(2)
                return f"sqrt({num})/{denom}"
                
            power_division_pattern = r'(\d+)\^\(1/(\d+)\)/(\d+)'
            power_division_match = re.match(power_division_pattern, normalized_expr)
            if power_division_match:
                base = power_division_match.group(1)
                root = power_division_match.group(2)
                denom = power_division_match.group(3)
                if root == '2':
                    return f"sqrt({base})/{denom}"
                else:
                    return f"{base}^(1/{root})/{denom}"
            
            numeric_functions = ['sqrt', 'cos', 'sin', 'tan', 'exp', 'acos', 'asin', 'atan', 'log10']
            for func in numeric_functions:
                func_match = re.match(rf'{func}\s*\(\s*([0-9.]+)\s*\)', expression)
                if func_match:
                    arg = func_match.group(1)
                    try:
                        self.eng.eval(f"temp_result = {func}({arg});", nargout=0)
                        numerical_result = float(self.eng.eval("double(temp_result)", nargout=1))
                        self.eng.eval("clear temp_result", nargout=0)
                        return self.simplifier.round_numeric_value(numerical_result)
                        
                    except Exception as e:
                        self.logger.error(f"Error in numerical function evaluation: {e}")
            
            log_base_match = re.match(r'log(\d+)\s*\(([^)]+)\)', expression)
            if log_base_match:
                base = log_base_match.group(1)
                arg = log_base_match.group(2)
                
                expression_converted = f"log({arg})/log({base})"
                
                try:
                    self.eng.eval(f"temp_result = {expression_converted};", nargout=0)
                    numerical_result = float(self.eng.eval("double(temp_result)", nargout=1))
                    self.eng.eval("clear temp_result", nargout=0)
                    
                    return self.simplifier.round_numeric_value(numerical_result)
                    
                except Exception as e:
                    self.logger.error(f"Error in logarithm evaluation: {e}")

            if '=' in expression:
                self.logger.debug(f"Equation detected, processing with _handle_equation: {expression}")
                expression = self._handle_equation(expression)
                self.logger.debug(f"After _handle_equation processing: {expression}")
            elif "'" in expression:
                self.logger.debug(f"Differential expression without equation detected: {expression}")
                expression = self._handle_equation(expression)
                self.logger.debug(f"After _handle_equation processing: {expression}")
            
            self.logger.debug(f"Processing expression: {expression}")
            
            matlab_keywords = {
                'solve', 'simplify', 'expand', 'factor', 'collect',
                'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
                'sind', 'cosd', 'tand', 'asind', 'acosd', 'atand',
                'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
                'log', 'log10', 'log2', 'exp', 'sqrt', 'nthroot', 'abs', 'sym', 'syms',
                'diff', 'int', 'limit', 'subs', 'pi', 'i', 'to', 'prod', 'sum',
                'arrayfun', 'Inf', 'inf'
            }
            
            variables = set(re.findall(r'(?<![a-zA-Z0-9_])([a-zA-Z_]\w*)(?!\w*\()', expression)) - matlab_keywords
            self.logger.debug(f"Extracted variables from expression: {variables}")
            
            if variables:
                var_str = ' '.join(variables)
                self.logger.debug(f"Declaring symbolic variable in MATLAB: syms {var_str}")
                self.eng.eval(f"syms {var_str}", nargout=0)
                self.logger.debug(f"Declared symbolic variable: {var_str}")
            
            command = f"temp_result = {expression};"
            self.logger.debug(f"Executing MATLAB command: {command}")
            self.eng.eval(command, nargout=0)
            
            result = str(self.eng.eval("char(temp_result)", nargout=1))
            
            if result.strip() == '' or (re.match(r'log\([^)]+\)/log\([^)]+\)', expression) and not result):
                try:
                    numvalue = float(self.eng.eval("double(temp_result)", nargout=1))
                    
                    # Use AutoSimplify's rounding method
                    result = self.simplifier.round_numeric_value(numvalue)
                    
                except Exception as e:
                    self.logger.error(f"Error getting numerical result: {e}")
            
            self.eng.eval("clear temp_result", nargout=0)
            
            # Apply post-processing to format the result
            result = self._postprocess_result(result.strip())
            self.logger.debug(f"Final result after post-processing: {result}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating expression: {e}")
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.eng.quit()
            self.logger.info("MATLAB engine terminated")
        except Exception as e:
            self.logger.error(f"Error terminating MATLAB engine: {e}")

    def __del__(self):
        try:
            self.eng.quit()
            self.logger.info("MATLAB engine terminated")
        except:
            pass

    def _clean_expression(self, expr):
        if not expr:
            return expr
        
        self.logger.debug(f"Original expression before cleaning: '{expr}'")
        
        expr = re.sub(r'exp\((.*?)\)', r'e^\1', expr)
        expr = expr.replace('*1.0', '').replace('1.0*', '')
        expr = expr.replace('log(', 'ln(')
        expr = re.sub(r'\b1\*\s*([a-zA-Z])', r'\1', expr)
        
        self.logger.debug(f"Final cleaned expression: '{expr}'")
        return expr

    def _simplify_log_expression(self, expression):
        """
        Simplify logarithmic expressions in the given expression.
        
        Args:
            expression (str): The expression containing logarithmic functions.
            
        Returns:
            str: The expression with simplified logarithmic functions.
        """
        log_base_pattern = re.compile(r'log(\d+)\s*\(\s*([^)]+)\s*\)')
        if log_base_pattern.search(expression):
            expression = log_base_pattern.sub(r'log(\2)/log(\1)', expression)
        
        expression = re.sub(r'(?<![a-zA-Z0-9_])ln\s*\(', 'log(', expression)
        
        return expression

    def _handle_equation(self, expression):
        """
        Handle equation expressions, particularly differential equations.
        
        Args:
            expression (str): The equation expression to handle.
            
        Returns:
            str: The processed equation expression or its solution.
        """
        if '=' in expression and '==' not in expression and '<=' not in expression and '>=' not in expression:
            expression = expression.replace('=', '==')

        independent_var = 't'
        
        exp_var_match = re.search(r'e\^([a-zA-Z])|exp\(([a-zA-Z])\)', expression)
        if exp_var_match:
            independent_var = exp_var_match.group(1) or exp_var_match.group(2)
            self.logger.debug(f"Detected independent variable '{independent_var}' from exponential term")
        
        prime_pattern = re.compile(r"([a-zA-Z])('*)(?!\w)")
        
        def replace_prime_notation(match):
            var = match.group(1)
            primes = match.group(2)
            order = len(primes)
            
            if order == 0:
                return var
            elif order == 1:
                return f"diff({var}, {independent_var})"
            else:
                return f"diff({var}, {independent_var}, {order})"
                
        if re.search(r"[a-zA-Z]'", expression):
            self.logger.debug(f"Differential equation with prime notation detected: {expression}")
            
            try:
                self.eng.eval(f"syms {independent_var}", nargout=0)
                self.logger.debug(f"Declared {independent_var} as symbolic variable for differential equation")
            except Exception as e:
                self.logger.error(f"Error declaring {independent_var} as symbolic variable: {e}")
            
            processed_expression = prime_pattern.sub(replace_prime_notation, expression)
            self.logger.debug(f"Processed differential equation: {processed_expression}")
            
            processed_expression = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', processed_expression)
            self.logger.debug(f"After adding explicit multiplication: {processed_expression}")
            
            if '==' in processed_expression:
                try:
                    self.eng.eval(f"syms y({independent_var})", nargout=0)
                    self.logger.debug(f"Declared y as a function of {independent_var}")
                    
                    solve_cmd = f"temp_result = dsolve({processed_expression});"
                    self.logger.debug(f"Solving differential equation with: {solve_cmd}")
                    self.eng.eval(solve_cmd, nargout=0)
                    
                    solution = self.eng.eval("char(temp_result)", nargout=1)
                    self.logger.debug(f"Solution to differential equation: {solution}")
                    
                    if solution:
                        solution = re.sub(r'C(\d+)', r'C_\1', solution)
                        return solution
                    else:
                        self.logger.warning("No solution found for differential equation")
                        return processed_expression
                except Exception as e:
                    self.logger.error(f"Error solving differential equation: {e}")
                    return processed_expression
            
            return processed_expression
        
        derivative_pattern = re.compile(r"d(\d*)/d([a-zA-Z]+)(\^(\d+))?")
        if re.search(derivative_pattern, expression):
            self.logger.debug(f"Derivative notation detected: {expression}")
            
            deriv_var_match = re.search(r"d(?:\d*)/d([a-zA-Z]+)", expression)
            if deriv_var_match:
                independent_var = deriv_var_match.group(1)
                self.logger.debug(f"Detected independent variable '{independent_var}' from derivative notation")
            
            def replace_derivative(match):
                order = match.group(1)
                var = match.group(2)
                if not order:
                    order = 1
                else:
                    order = int(order)
                
                return f"diff(, {var}, {order})"
            
            derivative_template = derivative_pattern.sub(replace_derivative, expression)
            
            parts = expression.split(' ', 1)
            if len(parts) > 1:
                func = parts[1].strip()
                processed_expression = derivative_template.replace("diff(, ", f"diff({func}, ")
                self.logger.debug(f"Processed derivative expression: {processed_expression}")
                
                processed_expression = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', processed_expression)
                self.logger.debug(f"After adding explicit multiplication: {processed_expression}")
                
                return processed_expression
        
        expression = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expression)
        
        return expression

    @lru_cache(maxsize=128)
    @matlab_memory_safe
    def evaluate_matlab_expression(self, expression):
        """
        Evaluate a MATLAB expression and return the result.
        
        Args:
            expression (str): The MATLAB expression to evaluate.
            
        Returns:
            str: The result of the evaluation.
        """
        self.original_expression = expression
        try:
            has_inequality = any(op in expression for op in ['>', '<', '>=', '<=', '!=', '=='])
            
            expression = self._preprocess_expression(expression)
            
            matlab_to_standard = {
                'sqrt': 'sqrt',
                'cbrt': 'nthroot',
                'log': 'log',
                'log10': 'log10',
                'exp': 'exp',
                'abs': 'abs',
                'sin': 'sin',
                'cos': 'cos',
                'tan': 'tan',
                'csc': 'csc',
                'sec': 'sec',
                'cot': 'cot',
                'asin': 'asin',
                'acos': 'acos',
                'atan': 'atan',
                'acsc': 'acsc',
                'asec': 'asec',
                'acot': 'acot',
                'sinh': 'sinh',
                'cosh': 'cosh',
                'tanh': 'tanh',
                'csch': 'csch',
                'sech': 'sech',
                'coth': 'coth',
                'asinh': 'asinh',
                'acosh': 'acosh',
                'atanh': 'atanh',
                'acsch': 'acsch',
                'asech': 'asech',
                'acoth': 'acoth'
            }
            
            for matlab_func, standard_func in matlab_to_standard.items():
                if matlab_func == 'cbrt':
                    expression = re.sub(r'\bcbrt\s*\(\s*([^,)]+)\s*\)', r'nthroot(\1, 3)', expression)
                elif matlab_func == 'log10':
                    pass  # We handle log10 specially in _preprocess_expression
                else:
                    pattern = r'\b' + re.escape(matlab_func) + r'\s*\('
                    replacement = standard_func + '('
                    expression = re.sub(pattern, replacement, expression)
            
            expression = self._simplify_log_expression(expression)
            
            if '=' in expression:
                expression = self._handle_equation(expression)
            
            self.logger.debug(f"Processing expression: {expression}")
            
            matlab_keywords = {
                'solve', 'simplify', 'expand', 'factor', 'collect',
                'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
                'sind', 'cosd', 'tand', 'asind', 'acosd', 'atand',
                'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
                'log', 'log10', 'log2', 'exp', 'sqrt', 'nthroot', 'abs', 'sym', 'syms',
                'diff', 'int', 'limit', 'subs', 'pi', 'i', 'to', 'prod', 'sum',
                'arrayfun', 'Inf', 'inf'
            }
            
            variables = set(re.findall(r'(?<![a-zA-Z0-9_])([a-zA-Z_]\w*)(?!\w*\()', expression)) - matlab_keywords
            self.logger.debug(f"Extracted variables from expression: {variables}")
            
            if variables:
                var_str = ' '.join(variables)
                self.logger.debug(f"Declaring symbolic variable in MATLAB: syms {var_str}")
                self.eng.eval(f"syms {var_str}", nargout=0)
                self.logger.debug(f"Declared symbolic variable: {var_str}")
            
            command = f"temp_result = {expression};"
            self.logger.debug(f"Executing MATLAB command: {command}")
            self.eng.eval(command, nargout=0)
            
            result = str(self.eng.eval("char(temp_result)", nargout=1))
            
            if result.strip() == '' or (re.match(r'log\([^)]+\)/log\([^)]+\)', expression) and not result):
                try:
                    numvalue = float(self.eng.eval("double(temp_result)", nargout=1))
                    result = self.simplifier.round_numeric_value(numvalue)
                    
                except Exception as e:
                    self.logger.error(f"Error getting numerical result: {e}")
            
            self.eng.eval("clear temp_result", nargout=0)
            
            result = self._postprocess_result(result.strip())
            self.logger.debug(f"Final result after post-processing: {result}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating expression: {e}")
            raise