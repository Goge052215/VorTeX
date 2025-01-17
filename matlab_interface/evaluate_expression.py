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

Version: 1.0.2
Last Updated: 2025.1
'''

import logging
import re
from functools import lru_cache
from matlab_interface.auto_simplify import AutoSimplify
from latex_pack.shortcut import ExpressionShortcuts

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
        # self.simplifier = AutoSimplify(eng)  # Fixing the auto-simplifier
        self._compile_patterns()
        self._initialize_workspace()
        self._symbolic_vars = set()

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
        self.trig_patterns = {
            re.compile(r'\bsind\s*\('): 'sind(',
            re.compile(r'\bcosd\s*\('): 'cosd(',
            re.compile(r'\btand\s*\('): 'tand(',
            re.compile(r'\barcsind\s*\('): 'asind(',
            re.compile(r'\barccosd\s*\('): 'acosd(',
            re.compile(r'\barctand\s*\('): 'atand('
        }
        self.log_e_pattern = re.compile(r'log\s*\(\s*E\s*,\s*([^,)]+)\)')
        self.log_base_pattern = re.compile(r'log\s*\(\s*(\d+)\s*,\s*([^,)]+)\)')
        self.mul_pattern = re.compile(r'(\d)([a-zA-Z])')

    def _initialize_workspace(self):
        """
        Initialize the MATLAB workspace with necessary constants and symbolic variables.
        """
        init_cmd = """
        pi = pi;      % Use MATLAB's built-in pi constant
        e = exp(1);   % Define e
        syms x;       % Define x as symbolic
        """
        self.eng.eval(init_cmd, nargout=0)
        self.logger.debug("Initialized e and pi in MATLAB workspace")

    def _preprocess_expression(self, expression):
        """
        Preprocess the expression before MATLAB evaluation.
        
        Args:
            expression (str): The input expression.
            
        Returns:
            str: The preprocessed expression.
        """
        original_expr = expression

        expression = ExpressionShortcuts.convert_sum_prod_expression(expression)
        expression = self.ln_pattern.sub('log(', expression)
        
        for trig_regex, degree_func in self.trig_patterns.items():
            expression = trig_regex.sub(degree_func, expression)
        
        expression = self.log_e_pattern.sub(r'log(\1)', expression)
        expression = self.log_base_pattern.sub(r'log(\2)/log(\1)', expression)
        expression = self.mul_pattern.sub(r'\1*\2', expression)
        
        if 'e^' in expression:
            expression = expression.replace('e^', 'exp(')
            if not expression.endswith(')'):
                expression += ')'
        
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
        if not result_str:
            return "0"
        
        # Try to convert to float and format large numbers
        try:
            # Remove trailing .000 if present
            result_str = result_str.rstrip('0').rstrip('.')
            float_val = float(result_str)
            
            # Always format very large or very small numbers in scientific notation
            if abs(float_val) > 1e10 or abs(float_val) < 1e-10:
                formatted = f"{float_val:.6e}"
                # Ensure consistent spacing in scientific notation
                parts = formatted.split('e')
                if len(parts) == 2:
                    return f"{parts[0]} e{parts[1]}"
                return formatted
            
            return f"{float_val:.6f}".rstrip('0').rstrip('.')
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
        
        symbolizations = {
            'exp(': 'e^',
            'asin(': 'arcsin(',
            'acos(': 'arccos(',
            'atan(': 'arctan(',
            'log(': 'ln('
        }
        
        trig_funcs = ['sin', 'cos', 'tan', 'csc', 'sec', 'cot']
        for func in trig_funcs:
            pattern = f'{func}\\(([^()]*(?:\\([^()]*\\)[^()]*)*)\\)'
            matches = list(re.finditer(pattern, result_str))
            for match in reversed(matches):
                full_match = match.group(0)
                arg = match.group(1)
                if '(' not in arg and ')' not in arg:
                    result_str = result_str.replace(full_match, f'{func}({arg})')
        
        for old, new in symbolizations.items():
            result_str = result_str.replace(old, new)
        
        if '/' in result_str:
            result_str = re.sub(r'\s+', '', result_str)
            result_str = result_str.replace(')(', ')*(')
        
        if result_str.count('(') < result_str.count(')'):
            result_str = result_str.rstrip(')')
        
        self.logger.debug(f"Symbolic result after postprocessing: {result_str}")
        return result_str

    @lru_cache(maxsize=128)
    def _extract_variables(self, expression):
        """
        Extract variables from the expression for MATLAB evaluation.
        
        Args:
            expression (str): The input expression.
        
        Returns:
            list: A list of variable names.
        """
        if not isinstance(expression, str):
            self.logger.error("Expression is not a string")
            raise TypeError("Expression must be a string")
        
        expression_clean = re.sub(
            r'd\^?\d*/d[a-zA-Z]+|\b(?:sin|cos|tan|log|exp|sqrt|abs|sind|cosd|tand)\b',
            '',
            expression
        )
        variables = set(re.findall(r'\b[A-Za-z]+\b', expression_clean))
        
        reserved_keywords = {
            'int', 'diff', 'syms', 'log', 'sin', 'cos', 'tan', 
            'exp', 'sqrt', 'abs', 'sind', 'cosd', 'tand', 'symsum', 
            'prod', 'solve', 'inf', 'Inf'
        }
        variables = {v for v in variables if v.lower() not in {k.lower() for k in reserved_keywords}}
        
        self.logger.debug(f"Extracted variables from expression: {variables}")
        return sorted(variables)

    @lru_cache(maxsize=128)
    def _declare_symbolic_variable(self, var):
        """
        Declare a variable as symbolic in MATLAB.
        
        Args:
            var (str): Variable name.
        """
        if var in self._symbolic_vars:
            return
        declaration_cmd = f"syms {var}"
        self.logger.debug(f"Declaring symbolic variable in MATLAB: {declaration_cmd}")
        self.eng.eval(declaration_cmd, nargout=0)
        self._symbolic_vars.add(var)
        self.logger.debug(f"Declared symbolic variable: {var}")

    def _format_exponential(self, expr_str):
        def replace_exp(match):
            content = match.group(1)
            if '+' in content or '-' in content or '*' in content or '/' in content:
                return f"e^({content})"
            return f"e^{content}"

        pattern = r'exp\(((?:[^()]+|\([^()]*\))*)\)'
        return re.sub(pattern, replace_exp, expr_str)

    def _simplify_log_expression(self, expr_str):
        """Simplify logarithmic expressions."""
        try:
            log_pattern = r'log\(([\w\d\+\-\*\/\(\)]+)\)/log\(([\d\.]+)\)'
            
            def log_replacement(match):
                numerator = match.group(1)  # The expression inside first log
                base = match.group(2)       # The base number
                
                # Handle special cases
                if base == '10':
                    return f'log10({numerator})'
                elif base == '2':
                    return f'log2({numerator})'
                elif base == 'e':
                    return f'ln({numerator})'
                else:
                    return f'log{base}({numerator})'  # Convert to logn(x) format
            
            # Apply the general log simplification
            expr_str = re.sub(log_pattern, log_replacement, expr_str)
            
            # Also handle MATLAB's fraction format
            fraction_pattern = r'\((\d+)\*log\(([\w\d\+\-\*\/\(\)]+)\)\)/(\d+)'
            
            def fraction_replacement(match):
                num = int(match.group(1))
                expr = match.group(2)
                denom = int(match.group(3))
                
                ratio = num / denom
                if abs(ratio - round(ratio)) < 1e-10:
                    base = round(ratio)
                    return f'log{base}({expr})'
                else:
                    return f'log({expr})/log({num}/{denom})'
            
            expr_str = re.sub(fraction_pattern, fraction_replacement, expr_str)
            
            return expr_str
        except Exception as e:
            self.logger.error(f"Error simplifying log expression: {e}")
            return expr_str

    def evaluate_matlab_expression(self, expression):
        """Evaluate a MATLAB expression and return the result."""
        try:
            expression = self._preprocess_expression(expression)
            
            self.logger.debug(f"Processing expression: {expression}")
            
            matlab_keywords = {
                'solve', 'simplify', 'expand', 'factor', 'collect',
                'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
                'sind', 'cosd', 'tand', 'asind', 'acosd', 'atand',
                'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
                'log', 'log10', 'log2', 'exp', 'sqrt', 'abs', 'sym', 'syms',
                'diff', 'int', 'limit', 'subs', 'pi', 'i', 'to', 'prod', 'sum',
                'arrayfun', 'Inf', 'inf'
            }
            
            variables = set(re.findall(r'(?<![a-zA-Z])([a-zA-Z_]\w*)(?!\w*\()', expression)) - matlab_keywords
            self.logger.debug(f"Extracted variables from expression: {variables}")
            
            if variables:
                var_str = ' '.join(variables)
                self.logger.debug(f"Declaring symbolic variable in MATLAB: syms {var_str}")
                self.eng.eval(f"syms {var_str}", nargout=0)
                self.logger.debug(f"Declared symbolic variable: {var_str}")
            
            command = f"temp_result = {expression};"
            self.logger.debug(f"Executing MATLAB command: {command}")
            self.eng.eval(command, nargout=0)
            
            # For limit evaluations
            if expression.startswith('limit('):
                try:
                    # Try to evaluate the limit numerically
                    self.eng.eval("temp_result = double(temp_result);", nargout=0)
                    result = str(self.eng.eval("num2str(temp_result, '%.3f')", nargout=1))
                    if result == 'Inf':
                        result = '∞'
                    elif result == '-Inf':
                        result = '-∞'
                except:
                    # If numerical evaluation fails, return symbolic result
                    result = str(self.eng.eval("char(temp_result)", nargout=1))
                    result = self._format_exponential(result)
                    result = self._simplify_log_expression(result)
            else:
                # Check if this is an indefinite integral
                is_indefinite_integral = 'int(' in expression and expression.count(',') == 2
                
                if expression.startswith('solve('):
                    self.eng.eval("temp_result = vpa(temp_result, 4);", nargout=0)
                    num_solutions = self.eng.eval("length(temp_result)", nargout=1)
                    
                    solutions = []
                    for i in range(int(num_solutions)):
                        solution = self.eng.eval(f"char(temp_result({i+1}))", nargout=1)
                        solution = self._format_exponential(solution)
                        solution = self._simplify_log_expression(solution)
                        solutions.append(f"x{i+1} = {solution}")
                    
                    result = '\n'.join(solutions)
                elif is_indefinite_integral:
                    # Handle indefinite integral result symbolically
                    self.eng.eval("temp_result = simplify(temp_result);", nargout=0)
                    result = str(self.eng.eval("char(temp_result)", nargout=1))
                    result = self._format_exponential(result)
                    result = self._simplify_log_expression(result)
                    if not result.endswith('+ C'):
                        result += ' + C'
                else:
                    is_numeric = self.eng.eval("~isa(temp_result, 'sym') && isnumeric(temp_result)", nargout=1)
                    has_variables = self.eng.eval("isa(temp_result, 'sym') && ~isempty(symvar(temp_result))", nargout=1)
                    
                    if is_numeric:
                        result = str(self.eng.eval("num2str(double(temp_result), '%.3f')", nargout=1))
                    elif has_variables:
                        self.eng.eval("temp_result = simplify(temp_result);", nargout=0)
                        result = str(self.eng.eval("char(temp_result)", nargout=1))
                        result = self._format_exponential(result)
                        result = self._simplify_log_expression(result)
                    else:
                        # For other cases, try to evaluate numerically
                        try:
                            self.eng.eval("temp_result = double(vpa(temp_result, 4));", nargout=0)
                            result = str(self.eng.eval("num2str(temp_result, '%.3f')", nargout=1))
                        except:
                            # If numerical evaluation fails, return symbolic result
                            result = str(self.eng.eval("char(temp_result)", nargout=1))
                            result = self._format_exponential(result)
                            result = self._simplify_log_expression(result)
            
            # Clean up workspace
            self.eng.eval("clear temp_result", nargout=0)
            
            return result.strip()
            
        except Exception as e:
            self.logger.error(f"Unexpected Error: {str(e)}")
            return str(e)

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

    def evaluate(self, expression: str) -> str:
        """Evaluate a mathematical expression using MATLAB."""
        try:
            self.logger.debug(f"Processing expression: {expression}")
            
            variables = self._extract_variables(expression)
            self.logger.debug(f"Extracted variables from expression: {variables}")
            
            for var in variables:
                self.logger.debug(f"Declaring symbolic variable in MATLAB: syms {var}")
                self.eng.eval(f"syms {var}", nargout=0)
                self.logger.debug(f"Declared symbolic variable: {var}")
            
            command = f"temp_result = {expression};"
            self.logger.debug(f"Executing MATLAB command: {command}")
            self.eng.eval(command, nargout=0)
            
            result = self.eng.eval("char(temp_result)", nargout=1)
            self.logger.debug(f"Raw result from MATLAB: '{result}'")
            
            # Preserve e^x notation
            if 'exp(' in result:
                result = result.replace('exp(', 'e^').rstrip(')')
            elif 'e^' in result:
                result = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating expression: {e}")
            raise