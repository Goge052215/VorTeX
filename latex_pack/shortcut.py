import re
import logging

class ExpressionShortcuts:
    """
    A class containing mappings for mathematical expression shortcuts to their LaTeX equivalents.
    """
    
    DERIVATIVE_SHORTCUTS = {
        'd/dx': r'\frac{d}{dx}',
        'd/dy': r'\frac{d}{dy}',
        'd/dt': r'\frac{d}{dt}',
        'd2/dx2': r'\frac{d^2}{dx^2}',
        'd3/dx3': r'\frac{d^3}{dx^3}',
        'd4/dx4': r'\frac{d^4}{dx^4}',
        'd5/dx5': r'\frac{d^5}{dx^5}',
        'd6/dx6': r'\frac{d^6}{dx^6}',
        'd7/dx7': r'\frac{d^7}{dx^7}',
        'd8/dx8': r'\frac{d^8}{dx^8}',
        'd9/dx9': r'\frac{d^9}{dx^9}',
        'd10/dx10': r'\frac{d^{10}}{dx^{10}}',

        'd2/dy2': r'\frac{d^2}{dy^2}',
        'd3/dy3': r'\frac{d^3}{dy^3}',
        'd4/dy4': r'\frac{d^4}{dy^4}',
        'd5/dy5': r'\frac{d^5}{dy^5}',

        'd2/dt2': r'\frac{d^2}{dt^2}',
        'd3/dt3': r'\frac{d^3}{dt^3}',
        'd4/dt4': r'\frac{d^4}{dt^4}',
        'd5/dt5': r'\frac{d^5}{dt^5}',

        'dx': r'\frac{d}{dx}',
        'dy': r'\frac{d}{dy}',
        'dt': r'\frac{d}{dt}',
        'd2x': r'\frac{d^2}{dx^2}',
        'd2y': r'\frac{d^2}{dy^2}',
        'd2t': r'\frac{d^2}{dt^2}',
        'd3x': r'\frac{d^3}{dx^3}',
        'd3y': r'\frac{d^3}{dy^3}',
        'd3t': r'\frac{d^3}{dt^3}'
    }
    
    COMBINATORIAL_SHORTCUTS = {
        'binom': 'nchoosek',
        'nCr': 'nchoosek',
        'choose': 'nchoosek'
    }
    
    INTEGRAL_SHORTCUTS = {
        'int': 'int',
        'int (a to b)': r'int_{a}^{b}',
        'integral': 'int',
        'iint': r'\iint',
        'iiint': r'\iiint',
        'oint': r'\oint',
    }
    
    FUNCTION_SHORTCUTS = {
        'sqrt': r'\sqrt',
        'root': r'\sqrt',
        'abs': r'\left|#\right|',
        'sin': r'sin',
        'cos': r'cos',
        'tan': r'tan',
        'sind': r'sind',
        'cosd': r'cosd',
        'tand': r'tand',
        'csc': r'csc',
        'sec': r'sec',
        'cot': r'cot',
        'arcsin': r'asin',
        'arccos': r'acos',
        'arctan': r'atan',
        'arccsc': r'acsc',
        'arcsec': r'asec',
        'arccot': r'acot',
        'asin': r'asin',
        'acos': r'acos',
        'atan': r'atan',
        'ln': r'ln',
        'lg': r'log_{10}',
        'log': r'log',
        'log10': r'log_{10}',
        'log2': r'log_{2}',
        'logn': r'log_{n}',
        'sinh': r'sinh',
        'cosh': r'cosh',
        'tanh': r'tanh',
        'csch': r'csch',
        'sech': r'sech',
        'coth': r'coth',
        'arcsinh': r'asinh',
        'arccosh': r'acosh',
        'arctanh': r'atanh',
        'arccsch': r'acsch',
        'arcsech': r'asech',
        'arccoth': r'acoth',
        'asinh': r'asinh',
        'acosh': r'acosh',
        'atanh': r'atanh',
    }
    
    FRACTION_SHORTCUTS = {
        '//': r'\frac{#}{#}',
    }
    
    GREEK_SHORTCUTS = {
        'alpha': 'alpha',
        'beta': 'beta',
        'gamma': 'gamma',
        'delta': 'delta',
        'epsilon': 'epsilon',
        'zeta': 'zeta',
        'eta': 'eta',
        'theta': 'theta',
        'iota': 'iota',
        'kappa': 'kappa',
        'lambda': 'lambda',
        'mu': 'mu',
        'nu': 'nu',
        'xi': 'xi',
        'pi': 'pi',
        'rho': 'rho',
        'sigma': 'sigma',
        'tau': 'tau',
        'upsilon': 'upsilon',
        'phi': 'phi',
        'chi': 'chi',
        'psi': 'psi',
        'omega': 'omega',
    }
    
    OPERATOR_SHORTCUTS = {
        'sum (a to b)': r'\sum_{a}^{b}',
        'prod (a to b)': r'\prod_{a}^{b}',
        'lim (x to a)': r'\lim_{x \to a}',
        'lim (x to a+)': r'\lim_{x \to a^{+}}',
        'lim (x to a-)': r'\lim_{x \to a^{-}}',
        'to': r'\to',
        'rightarrow': r'\rightarrow',   
        'leftarrow': r'\leftarrow',
        'infty': r'\infty',
        'infinity': r'\infty',
    }
    
    EQUATION_SHORTCUTS = {
        '=': '=',
        '>=': r'\geq',
        '<=': r'\leq',
        '!=': r'\neq',
        '>>': r'\gg',
        '<<': r'\ll',
        'approx': r'\approx',
        'equiv': r'\equiv',
        'prop': r'\propto',
        'sim': r'\sim',
    }
    
    @classmethod
    def get_all_shortcuts(cls):
        all_shortcuts = {}
        all_shortcuts.update(cls.DERIVATIVE_SHORTCUTS)
        all_shortcuts.update(cls.COMBINATORIAL_SHORTCUTS)
        all_shortcuts.update(cls.INTEGRAL_SHORTCUTS)
        all_shortcuts.update(cls.FUNCTION_SHORTCUTS)
        all_shortcuts.update(cls.FRACTION_SHORTCUTS)
        all_shortcuts.update(cls.GREEK_SHORTCUTS)
        all_shortcuts.update(cls.OPERATOR_SHORTCUTS)
        all_shortcuts.update(cls.EQUATION_SHORTCUTS)
        return all_shortcuts
    
    @classmethod
    def convert_shortcut(cls, text):
        result = text
        
        if text.startswith('d') and ('/' in text or text[1:2].isdigit()):
            parts = text.split(' ', 1)
            if len(parts) == 2:
                derivative_part, function_part = parts
                
                if '/' in derivative_part:
                    order_match = re.match(r'd(\d*)/d([xyzt])(\d*)', derivative_part)
                    if order_match:
                        order = order_match.group(1) or '1'
                        var = order_match.group(2)
                        power = order_match.group(3) or order  # Use same number for denominator power
                        return f"\\frac{{d^{order}}}{{d{var}^{power}}} {function_part}"
                else:
                    order_match = re.match(r'd(\d*)([xyzt])', derivative_part)
                    if order_match:
                        order = order_match.group(1) or '1'
                        var = order_match.group(2)
                        return f"\\frac{{d^{order}}}{{d{var}^{order}}} {function_part}"
                    order_match = re.match(r'd(\d*)([xyz])', derivative_part)
                    if order_match:
                        order = order_match.group(1) or '1'
                        var = order_match.group(2)
                        return f"\\frac{{d^{order}}}{{d{var}^{order}}} {function_part}"

        result = cls.convert_exponential_expression(result)
        result = cls.convert_integral_expression(result)
        result = cls.convert_limit_expression(result)
        result = cls._convert_logarithms(result)
        result = cls.convert_combinatorial_expression(result)
        result = cls.convert_sum_prod_expression(result)
        
        shortcuts = {k: v for k, v in cls.get_all_shortcuts().items() if '#' not in v}
        
        if shortcuts:
            pattern = '|'.join(map(re.escape, sorted(shortcuts.keys(), key=len, reverse=True)))
            result = re.sub(pattern, lambda m: shortcuts[m.group()], result)
        
        return result.replace('\\', '')
    
    @classmethod
    def convert_integral_expression(cls, text):
        """
        Convert integral expressions to a format suitable for parsing.
        
        Args:
            text (str): Input text containing integral expressions.
            
        Returns:
            str: Text with integral expressions converted.
        """
        definite_integral_pattern = r'int\s*\(\s*([^\s]+?)\s*to\s*([^\s]+?)\s*\)\s*([^\s]+?)\s*d([a-zA-Z])'
        
        def replace_definite_integral(match):
            lower = match.group(1).strip()
            upper = match.group(2).strip()
            expr = match.group(3).strip()
            var = match.group(4).strip()
            
            # Handle function expressions in limits
            if 'ln' in lower:
                lower = lower.replace('ln', 'log')
            if 'ln' in upper:
                upper = upper.replace('ln', 'log')
            
            return f'int({expr}, {var}, {lower}, {upper})'
        
        text = re.sub(definite_integral_pattern, replace_definite_integral, text)
        
        indefinite_integral_pattern = r'int\s+([^\s]+?)\s+d([a-zA-Z])'
        
        def replace_indefinite_integral(match):
            expr = match.group(1).strip()
            var = match.group(2).strip()
            
            if 'ln' in expr:
                expr = expr.replace('ln', 'log')
            
            return f'int({expr}, {var})'
        
        text = re.sub(indefinite_integral_pattern, replace_indefinite_integral, text)
        
        return text

    @staticmethod
    def _convert_logarithms(expr):
        """Convert different logarithm notations to MATLAB format."""
        expr = re.sub(r'lg\s*\((.*?)\)', r'log10(\1)', expr)
        expr = re.sub(r'ln\s*\((.*?)\)', r'log(\1)', expr)
        expr = re.sub(r'log(\d+)\s*\((.*?)\)', lambda m: f'log({m.group(2)})/log({m.group(1)})', expr)
        
        return expr

    @staticmethod
    def convert_combinatorial_expression(expr):
        """Convert combinatorial expressions to MATLAB format."""
        expr = re.sub(r'binom\s*\(([^,]+),([^)]+)\)', r'choose(\1,\2)', expr)
        expr = re.sub(r'(\w+|\d+|\([^)]+\))C(\w+|\d+|\([^)]+\))', r'choose(\1,\2)', expr)
        expr = re.sub(r'binom\s*\{([^}]+)\}\s*\{([^}]+)\}', r'choose(\1,\2)', expr)
        
        return expr
    
    @staticmethod
    def convert_sum_prod_expression(expr):
        """Convert sum and prod expressions to MATLAB format."""
        def extract_variable(expr_str):
            vars = re.findall(r'(?<![a-zA-Z])([a-zA-Z])(?![a-zA-Z])', expr_str)
            reserved = {'e', 'i', 'n', 'inf', 'Inf'}
            vars = [v for v in vars if v not in reserved]
            return vars[0] if vars else 'x'

        harmonic_pattern = r'sum\s*\(\s*(\d+)\s*to\s*(?:inf|Inf|(\d+))\s*\)\s*1/([a-zA-Z])'
        def replace_harmonic(match):
            start = int(match.group(1))
            end = match.group(2)
            var = match.group(3)
            
            if end is None:
                return "∞"
            else:
                if start == 1:
                    return f"double(log({end}) + 0.57721566490153286060)"
                else:
                    return f"double(log({end}) - log({start-1}))"

        if re.search(harmonic_pattern, expr):
            expr = re.sub(harmonic_pattern, replace_harmonic, expr)
            return expr

        finite_prod_pattern = r'prod\s*\(\s*([+-]?\d+|[+-]?inf)\s*to\s*([+-]?\d+|[+-]?inf)\s*\)\s*([^\n]+)'
        def replace_finite_prod(match):
            start = match.group(1).strip()
            end = match.group(2).strip()
            expr_part = match.group(3).strip()
            var = extract_variable(expr_part)
            
            if 'inf' in start.lower():
                start = '-Inf' if start.startswith('-') else 'Inf'
            if 'inf' in end.lower():
                end = '-Inf' if end.startswith('-') else 'Inf'
            
            return f"prod(arrayfun(@(k) subs({expr_part}, {var}, k), {start}:{end}))"
        
        inf_prod_pattern = r'prod\s*\(\s*([+-]?\d+|[+-]?inf)\s*to\s*(?:inf|Inf)\s*\)\s*([^\n]+)'
        def replace_infinite_prod(match):
            start = match.group(1).strip()
            expr_part = match.group(2).strip()
            var = extract_variable(expr_part)
            
            if 'inf' in start.lower():
                start = '-Inf' if start.startswith('-') else 'Inf'
            
            return f"symprod({expr_part}, {var}, {start}, Inf)"

        sum_pattern = r'sum\s*\(\s*([+-]?\d+|[+-]?inf)\s*to\s*(inf|Inf|[+-]?\d+)\s*\)\s*([^\n]+)'
        def replace_sum(match):
            start = match.group(1).strip()
            end = match.group(2).strip().lower()
            expr_part = match.group(3).strip()
            var = extract_variable(expr_part)
            
            if 'inf' in start.lower():
                start = '-Inf' if start.startswith('-') else 'Inf'
            if 'inf' in end:
                end = '-Inf' if end.startswith('-') else 'Inf'
            
            return f"symsum({expr_part}, {var}, {start}, {end})"
        
        expr = re.sub(inf_prod_pattern, replace_infinite_prod, expr)
        expr = re.sub(finite_prod_pattern, replace_finite_prod, expr)
        expr = re.sub(sum_pattern, replace_sum, expr)

        if 'Inf' in expr:
            expr = f"limit({expr}, x, Inf)"

        return expr

    @staticmethod
    def convert_limit_expression(expr):
        """Convert limit expressions to MATLAB format."""
        limit_pattern = r'(?i)lim\s*\(\s*([a-zA-Z])\s*to\s*' + \
                       r'(' + \
                       r'[-+]?\d*\.?\d+|' + \
                       r'[-+]?(?:inf(?:ty|inity)?)|' + \
                       r'[a-zA-Z][a-zA-Z0-9]*|' + \
                       r'(?:sin|cos|tan|csc|sec|cot|sinh|cosh|tanh|sech|csch|coth|arcsin|arccos|arctan|arcsec|arccsc|arccot|asinh|acosh|atanh|asech|acsch|acoth|ln|log|log\d+|sqrt|exp)\s*\([^)]+\)|' + \
                       r'e\^?[^,\s)]*' + \
                       r')([+-])?\s*\)\s*' + \
                       r'((?:[^()]+|\((?:[^()]+|\([^()]*\))*\))*)'
        
        def replace_limit(match):
            var, approach, side, function = match.groups()
            
            # Convert infinity variations
            if isinstance(approach, str) and re.match(r'(?i)inf(?:ty|inity)?', approach):
                approach = 'inf'
            elif isinstance(approach, str) and re.match(r'(?i)[+-]inf(?:ty|inity)?', approach):
                sign = approach[0]
                approach = f'{sign}inf'
            elif '(' in approach:
                approach = re.sub(r'arcsin\(', r'asin(', approach)
                approach = re.sub(r'arccos\(', r'acos(', approach)
                approach = re.sub(r'arctan\(', r'atan(', approach)
                approach = re.sub(r'arcsec\(', r'asec(', approach)
                approach = re.sub(r'arccsc\(', r'acsc(', approach)
                approach = re.sub(r'arccot\(', r'acot(', approach)
                approach = re.sub(r'arcsinh\(', r'asinh(', approach)
                approach = re.sub(r'arccosh\(', r'acosh(', approach)
                approach = re.sub(r'arctanh\(', r'atanh(', approach)
                approach = re.sub(r'arcsech\(', r'asech(', approach)
                approach = re.sub(r'arccsch\(', r'acsch(', approach)
                approach = re.sub(r'arccoth\(', r'acoth(', approach)
                approach = re.sub(r'ln\(', r'log(', approach)
                if re.match(r'log\d+\(', approach):
                    base = re.match(r'log(\d+)', approach).group(1)
                    arg = re.search(r'\((.*?)\)', approach).group(1)
                    approach = f'log({arg})/log({base})'

            elif approach.startswith('e^'):
                approach = f'exp({approach[2:]})'
            elif approach == 'e':
                approach = 'exp(1)'
            
            function = function.strip()
            function = re.sub(r'arcsin\(', r'asin(', function)
            function = re.sub(r'arccos\(', r'acos(', function)
            function = re.sub(r'arctan\(', r'atan(', function)
            function = re.sub(r'arcsec\(', r'asec(', function)
            function = re.sub(r'arccsc\(', r'acsc(', function)
            function = re.sub(r'arccot\(', r'acot(', function)
            function = re.sub(r'arcsinh\(', r'asinh(', function)
            function = re.sub(r'arccosh\(', r'acosh(', function)
            function = re.sub(r'arctanh\(', r'atanh(', function)
            function = re.sub(r'arcsech\(', r'asech(', function)
            function = re.sub(r'arccsch\(', r'acsch(', function)
            function = re.sub(r'arccoth\(', r'acoth(', function)
            function = re.sub(r'ln\(', r'log(', function)

            if re.search(r'log\d+\(', function):
                function = re.sub(r'log(\d+)\((.*?)\)', 
                                lambda m: f'log({m.group(2)})/log({m.group(1)})', 
                                function)
            if 'e^' in function:
                function = function.replace('e^', 'exp(') + ')'
            
            if side:
                if side == '+':
                    return f"limit({function}, {var}, {approach}, 'right')"
                elif side == '-':
                    return f"limit({function}, {var}, {approach}, 'left')"
            
            return f"limit({function}, {var}, {approach})"

        expr = re.sub(limit_pattern, replace_limit, expr)
        return expr

    @staticmethod
    def convert_exponential_expression(expr):
        pattern = r'\be\^([a-zA-Z0-9\+\-\*/\(\)]+)'
        replacement = r'exp(\1)'
        converted_expr = re.sub(pattern, replacement, expr)
        return converted_expr

    @classmethod
    def convert_equation(cls, text):
        """
        Convert equation expressions to LaTeX format.
        
        Args:
            text (str): Input text containing equation
            
        Returns:
            str: LaTeX formatted equation
        """
        result = text
        
        if '=' in result:
            parts = result.split('=')
            processed_parts = []
            
            for part in parts:
                # Process each side of the equation
                processed_part = cls.convert_shortcut(part.strip())
                processed_parts.append(processed_part)
            
            result = ' = '.join(processed_parts)
            
        for symbol, latex in cls.EQUATION_SHORTCUTS.items():
            if symbol in result:
                result = result.replace(symbol, latex)
        
        return result