import re
import logging

class ExpressionShortcuts:
    """
    A class containing mappings for mathematical expression shortcuts to their LaTeX equivalents.
    """
    
    # Derivative shortcuts
    DERIVATIVE_SHORTCUTS = {
        'd/dx': r'\frac{d}{dx}',
        'd/dy': r'\frac{d}{dy}',
        'd/dt': r'\frac{d}{dt}',
        'd2/dx2': r'\frac{d^2}{dx^2}',
        'd3/dx3': r'\frac{d^3}{dx^3}',
        'd4/dx4': r'\frac{d^4}{dx^4}',
        'd5/dx5': r'\frac{d^5}{dx^5}'
    }
    
    # Combinatorial shortcuts
    COMBINATORIAL_SHORTCUTS = {
        'binom': r'nchoosek',  # Convert binom to nchoosek for MATLAB
        'nCr': r'nchoosek(n,r)',      # Alternative notation
        'choose': r'nchoosek'  # Another common notation
    }
    
    # Integral shortcuts
    INTEGRAL_SHORTCUTS = {
        'int': 'int',
        'int (a to b)': r'int_{a}^{b}',
        'integral': 'int',
        'iint': r'\iint',  # Double integral
        'iiint': r'\iiint',  # Triple integral
        'oint': r'\oint',  # Contour integral
    }
    
    # Function shortcuts
    FUNCTION_SHORTCUTS = {
        'sqrt': r'\sqrt',
        'root': r'\sqrt',
        'abs': r'\left|#\right|',  # # will be replaced with the argument
        'sin': r'\sin',
        'cos': r'\cos',
        'tan': r'\tan',
        'csc': r'\csc',
        'sec': r'\sec',
        'cot': r'\cot',
        'arcsin': r'\arcsin',
        'arccos': r'\arccos',
        'arctan': r'\arctan',
        'ln': r'ln',
        'lg': r'log_{10}',  # base-10 logarithm
        'log': r'log',
        'log10': r'log_{10}',  # Explicit base-10 log
        'log2': r'log_{2}',    # Base-2 log
        'logn': r'log_{n}',    # Base-n log
    }
    
    # Fraction shortcuts
    FRACTION_SHORTCUTS = {
        '//': r'\frac{#}{#}',  # #'s will be replaced with numerator and denominator
    }
    
    # Greek letters
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
    
    # Operator shortcuts
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
    
    @classmethod
    def get_all_shortcuts(cls):
        """
        Get all shortcuts combined into a single dictionary.
        
        Returns:
            dict: Combined dictionary of all shortcuts
        """
        all_shortcuts = {}
        all_shortcuts.update(cls.DERIVATIVE_SHORTCUTS)
        all_shortcuts.update(cls.COMBINATORIAL_SHORTCUTS)
        all_shortcuts.update(cls.INTEGRAL_SHORTCUTS)
        all_shortcuts.update(cls.FUNCTION_SHORTCUTS)
        all_shortcuts.update(cls.FRACTION_SHORTCUTS)
        all_shortcuts.update(cls.GREEK_SHORTCUTS)
        all_shortcuts.update(cls.OPERATOR_SHORTCUTS)
        return all_shortcuts
    
    @classmethod
    def     convert_shortcut(cls, text):
        """
        Convert shortcuts in text to their LaTeX equivalents.
        
        Args:
            text (str): Input text containing shortcuts
            
        Returns:
            str: Text with shortcuts converted to LaTeX
        """
        result = text
        
        # Convert exponential expressions first
        result = cls.convert_exponential_expression(result)
        logging.debug(f"After exponential conversion: {result}")
        
        # Convert integral expressions (without adding backslash)
        result = cls.convert_integral_expression(result)
        logging.debug(f"After integral conversion: {result}")
        
        # Handle limits before other conversions
        result = cls.convert_limit_expression(result)
        logging.debug(f"After limit conversion: {result}")
        
        # Handle logarithms with different bases
        result = cls._convert_logarithms(result)
        logging.debug(f"After logarithm conversion: {result}")

        # Handle other shortcuts
        result = cls.convert_combinatorial_expression(result)
        logging.debug(f"After combinatorial conversion: {result}")
        
        result = cls.convert_sum_prod_expression(result)
        logging.debug(f"After sum and prod conversion: {result}")
        
        # Handle higher-order derivative notation (e.g., "d2/dx2 x^2")
        if text.startswith('d') and ('/' in text or text[1:2].isdigit()):
            parts = text.split(' ', 1)
            if len(parts) == 2:
                derivative_part, function_part = parts
                
                # Handle different derivative notations
                if '/' in derivative_part:
                    # Handle d/dx or d2/dx2 notation
                    order_match = re.match(r'd(\d*)/d([xyz])(\d*)', derivative_part)
                    if order_match:
                        order = order_match.group(1) or '1'
                        var = order_match.group(2)
                        result = f"\\frac{{d^{order}}}{{d{var}^{order}}} {function_part}"
                else:
                    # Handle d2x or dx notation
                    order_match = re.match(r'd(\d*)([xyz])', derivative_part)
                    if order_match:
                        order = order_match.group(1) or '1'
                        var = order_match.group(2)
                        result = f"\\frac{{d^{order}}}{{d{var}^{order}}} {function_part}"
                
                return result
        
        # Handle other shortcuts
        shortcuts = cls.get_all_shortcuts()
        for shortcut, latex in shortcuts.items():
            if shortcut in result and '#' not in latex:
                result = result.replace(shortcut, latex)
        
        return result
    
    @classmethod
    def convert_integral_expression(cls, text):
        """
        Convert integral expressions to a format suitable for parsing.
        
        Args:
            text (str): Input text containing integral expressions.
            
        Returns:
            str: Text with integral expressions converted.
        """
        # Convert definite integrals like 'int (a to b) expr dx' to 'int(expr, x, a, b)'
        definite_integral_pattern = r'int\s*\(([^)]+)\s*to\s*([^)]+)\)\s+(.+?)\s+d([a-zA-Z])'
        
        def replace_definite_integral(match):
            lower = match.group(1).strip()
            upper = match.group(2).strip()
            expr = match.group(3).strip()
            var = match.group(4).strip()
            return f'int({expr}, {var}, {lower}, {upper})'
        
        text = re.sub(definite_integral_pattern, replace_definite_integral, text)
        
        # Convert indefinite integrals like 'int expr dx' to 'int(expr, x)'
        indefinite_integral_pattern = r'int\s+(.+?)\s+d([a-zA-Z])'
        
        def replace_indefinite_integral(match):
            expr = match.group(1).strip()
            var = match.group(2).strip()
            return f'int({expr}, {var})'
        
        text = re.sub(indefinite_integral_pattern, replace_indefinite_integral, text)
        
        return text

    @staticmethod
    def _convert_logarithms(expr):
        """Convert different logarithm notations to MATLAB format."""
        # Convert lg(x) to log10(x)
        expr = re.sub(r'lg\s*\((.*?)\)', r'log10(\1)', expr)
        
        # Keep ln(x) as log(x) for natural logarithm
        expr = re.sub(r'ln\s*\((.*?)\)', r'log(\1)', expr)
        
        # Convert logN(x) to log(x)/log(N) for any base N
        expr = re.sub(r'log(\d+)\s*\((.*?)\)', lambda m: f'log({m.group(2)})/log({m.group(1)})', expr)
        
        return expr

    @staticmethod
    def convert_combinatorial_expression(expr):
        """Convert combinatorial expressions to MATLAB format."""
        # Convert nCr(n, k) or nCr to nchoosek(n, k)
        expr = re.sub(r'(\d+)C(\d+)', r'nchoosek(\1, \2)', expr)
        return expr
    
    @staticmethod
    def convert_sum_prod_expression(expr):
        """Convert sum and prod expressions to MATLAB format."""
        # Enhanced pattern to handle spaces and case insensitivity
        sum_pattern = r'(?i)sum\s*\(\s*(\d+)\s*to\s*(\d+)\s*\)\s*([^\n]+)'
        
        def replace_sum(match):
            start, end, function = match.groups()
            return f"symsum({function}, x, {start}, {end})"

        expr = re.sub(sum_pattern, replace_sum, expr)
        
        # Adjust prod pattern to use a loop or symbolic product
        prod_pattern = r'(?i)prod\s*\(\s*(\d+)\s*to\s*(\d+)\s*\)\s*([^\n]+)'
        
        def replace_prod(match):
            start, end, function = match.groups()
            # Use a loop or symbolic product function
            return f"prod(arrayfun(@(x) {function}, {start}:{end}))"

        expr = re.sub(prod_pattern, replace_prod, expr)
        
        return expr

    @staticmethod
    def convert_limit_expression(expr):
        """Convert limit expressions to MATLAB format."""
        # Pattern to match limit expressions including all function types
        limit_pattern = r'(?i)lim\s*\(\s*([a-zA-Z])\s*to\s*' + \
                       r'(' + \
                       r'[-+]?\d*\.?\d+|' + \
                       r'[-+]?(?:inf(?:ty|inity)?)|' + \
                       r'[a-zA-Z][a-zA-Z0-9]*|' + \
                       r'(?:sin|cos|tan|csc|sec|cot|arcsin|arccos|arctan|ln|log|log\d+|sqrt|exp)\s*\([^)]+\)|' + \
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
            # Handle functions in approach
            elif '(' in approach:
                # Convert trigonometric functions
                approach = re.sub(r'arcsin\(', r'asin(', approach)
                approach = re.sub(r'arccos\(', r'acos(', approach)
                approach = re.sub(r'arctan\(', r'atan(', approach)
                # Convert logarithms
                approach = re.sub(r'ln\(', r'log(', approach)
                if re.match(r'log\d+\(', approach):
                    base = re.match(r'log(\d+)', approach).group(1)
                    arg = re.search(r'\((.*?)\)', approach).group(1)
                    approach = f'log({arg})/log({base})'
            # Handle exponential e
            elif approach.startswith('e^'):
                approach = f'exp({approach[2:]})'
            elif approach == 'e':
                approach = 'exp(1)'
            
            # Format the function part
            function = function.strip()
            # Convert trigonometric functions
            function = re.sub(r'arcsin\(', r'asin(', function)
            function = re.sub(r'arccos\(', r'acos(', function)
            function = re.sub(r'arctan\(', r'atan(', function)
            # Convert logarithms
            function = re.sub(r'ln\(', r'log(', function)
            if re.search(r'log\d+\(', function):
                function = re.sub(r'log(\d+)\((.*?)\)', 
                                lambda m: f'log({m.group(2)})/log({m.group(1)})', 
                                function)
            # Convert exponentials
            if 'e^' in function:
                function = function.replace('e^', 'exp(') + ')'
            
            # Handle one-sided limits
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
        """Convert e^x to exp(x) to ensure correct MATLAB interpretation."""
        pattern = r'\be\^([a-zA-Z0-9\+\-\*/\(\)]+)'
        replacement = r'exp(\1)'
        converted_expr = re.sub(pattern, replacement, expr)
        return converted_expr