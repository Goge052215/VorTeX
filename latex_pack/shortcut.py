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
        'C': r'nchoosek',      # Alternative notation
        'choose': r'nchoosek'  # Another common notation
    }
    
    # Integral shortcuts
    INTEGRAL_SHORTCUTS = {
        'int': r'\int',
        'int (a to b)': r'\int_{a}^{b}',
        'integral': r'\int',
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
        'ln': r'\ln',
        'lg': r'\log_{10}',  # base-10 logarithm
        'log': r'\log',
        'log10': r'\log_{10}',  # Explicit base-10 log
        'log2': r'\log_{2}'     # Base-2 log
    }
    
    # Fraction shortcuts
    FRACTION_SHORTCUTS = {
        '//': r'\frac{#}{#}',  # #'s will be replaced with numerator and denominator
    }
    
    # Greek letters
    GREEK_SHORTCUTS = {
        'alpha': r'\alpha',
        'beta': r'\beta',
        'gamma': r'\gamma',
        'delta': r'\delta',
        'epsilon': r'\epsilon',
        'zeta': r'\zeta',
        'eta': r'\eta',
        'theta': r'\theta',
        'iota': r'\iota',
        'kappa': r'\kappa',
        'lambda': r'\lambda',
        'mu': r'\mu',
        'nu': r'\nu',
        'xi': r'\xi',
        'pi': r'\pi',
        'rho': r'\rho',
        'sigma': r'\sigma',
        'tau': r'\tau',
        'upsilon': r'\upsilon',
        'phi': r'\phi',
        'chi': r'\chi',
        'psi': r'\psi',
        'omega': r'\omega',
    }
    
    # Operator shortcuts
    OPERATOR_SHORTCUTS = {
        'sum (a to b)': r'\sum_{a}^{b}',
        'prod (a to b)': r'\prod_{a}^{b}',
        'lim': r'\lim',
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
    def convert_shortcut(cls, text):
        """
        Convert shortcuts in text to their LaTeX equivalents.
        
        Args:
            text (str): Input text containing shortcuts
            
        Returns:
            str: Text with shortcuts converted to LaTeX
        """
        result = text
        
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
        
        # Convert log(x) to log10(x) if not already handled
        expr = re.sub(r'(?<![\w])log\s*\((.*?)\)', r'log10(\1)', expr)
        
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
