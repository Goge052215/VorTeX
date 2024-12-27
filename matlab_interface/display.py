import re
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QFont

class Display:
    """
    A class to handle the formatting and display of mathematical expressions with proper symbols.
    """
    
    # Mapping of standard symbols to their mathematical counterparts
    SYMBOL_MAPPINGS = {
        '-': '−',          # Minus sign
        '+': '+',          # Plus sign remains the same
        '*': '×',          # Multiplication sign
        '/': '÷',          # Division sign
        '^': '^',          # Exponent remains '^'; handled separately for superscripts
        '=': '=',          # Equals sign remains the same
        '(': '(', ')': ')',
        '{': '{', '}': '}',
        '[': '[', ']': ']',
        'pi': 'π',         # Pi symbol
        'e': 'e',          # Euler's number remains the same
        'sqrt': '√',       # Square root symbol
        'ln': 'ln',        # Natural logarithm remains the same
        'log10': 'log₁₀',  # Logarithm base 10
        'exp': 'exp',      # Exponential function remains the same
        'sin': 'sin',      # Sine function remains the same
        'cos': 'cos',      # Cosine function remains the same
        'tan': 'tan',      # Tangent function remains the same
        'asin': 'arcsin',  # Arcsine function
        'acos': 'arccos',  # Arccosine function
        'atan': 'arctan',  # Arctangent function
        'abs': 'abs',      # Absolute value remains the same
        'lim': 'lim',      # Limit remains the same
        'sum': '∑',        # Summation symbol
        'prod': '∏',       # Product symbol
        'int': '∫',        # Integral symbol
        'infty': '∞',      # Infinity symbol
        'neq': '≠',        # Not equal
        'leq': '≤',        # Less than or equal to
        'geq': '≥',        # Greater than or equal to
        '<=': '≤',
        '>=': '≥',
        '!=': '≠',
        '<>': '≠',
        'rightarrow': '→', # Right arrow
        'leftarrow': '←',  # Left arrow
        # Add more mappings as needed
    }

    def __init__(self, label: QLabel, font_name: str = "Arial", font_size: int = 14, bold: bool = False):
        """
        Initialize the Display class with a QLabel instance.
        
        Args:
            label (QLabel): The label widget where expressions will be displayed.
            font_name (str): Name of the font to use.
            font_size (int): Size of the font.
            bold (bool): Whether the font is bold.
        """
        self.label = label
        self.font = QFont(font_name, font_size)
        if bold:
            self.font.setBold(True)
        self.label.setFont(self.font)
    
    def format_expression(self, expression: str) -> str:
        """
        Format the mathematical expression by replacing standard symbols with proper symbols.
        
        Args:
            expression (str): The original mathematical expression.
        
        Returns:
            str: The formatted expression with proper symbols.
        """
        formatted_expr = expression

        # Replace symbols based on the SYMBOL_MAPPINGS
        for key, value in self.SYMBOL_MAPPINGS.items():
            # Use word boundaries for specific words like 'pi'
            if re.fullmatch(r'\w+', key):
                formatted_expr = re.sub(rf'\b{re.escape(key)}\b', value, formatted_expr)
            else:
                # Replace all occurrences of the symbol
                formatted_expr = formatted_expr.replace(key, value)
        
        # Handle exponents: Convert '^n' to superscript characters if possible
        formatted_expr = self._replace_exponents(formatted_expr)
        
        return formatted_expr
    
    def _replace_exponents(self, expression: str) -> str:
        """
        Replace exponents in the expression with superscript characters.
        
        Args:
            expression (str): The expression with '^' for exponents.
        
        Returns:
            str: The expression with superscript characters.
        """
        # Regex to find exponents e.g., x^2, sin(x)^3, (a+b)^n
        pattern = r'\^(-?\d+)'
        matches = re.findall(pattern, expression)

        for match in matches:
            superscript = self._to_superscript(match)
            expression = re.sub(rf'\^{re.escape(match)}', superscript, expression)
        
        return expression
    
    def _to_superscript(self, number: str) -> str:
        """
        Convert a number string to its superscript representation.
        
        Args:
            number (str): The number to convert.
        
        Returns:
            str: The superscript representation of the number.
        """
        superscript_mapping = {
            '-': '⁻',
            '0': '⁰',
            '1': '¹',
            '2': '²',
            '3': '³',
            '4': '⁴',
            '5': '⁵',
            '6': '⁶',
            '7': '⁷',
            '8': '⁸',
            '9': '⁹',
        }
        return ''.join(superscript_mapping.get(char, char) for char in number)
    
    def display_expression(self, expression: str, bold: bool = False):
        """
        Format the expression and display it in the QLabel.
        
        Args:
            expression (str): The mathematical expression to display.
            bold (bool): Whether the displayed text is bold.
        """
        formatted = self.format_expression(expression)
        self.label.setText(formatted)
        self.label.setFont(self.font)
        if bold:
            self.label.setFont(QFont(self.font.family(), self.font.pointSize(), QFont.Bold))
    
    def display_result(self, result: str):
        """
        Format the result and display it in the QLabel with "Result: " prefix.
        
        Args:
            result (str): The result of the calculation to display.
        """
        formatted = self.format_expression(result)
        self.label.setText(f"Result: {formatted}")
        self.label.setFont(self.font)
