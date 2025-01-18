import re
import sympy as sy
import logging
import warnings

class SympyToMatlab:
    """
    DEPRECATED: This class is deprecated and will be removed in a future version.
    Use direct MATLAB expression evaluation or LaTeX parsing instead.
    """
    
    def __init__(self):
        warnings.warn(
            "SympyToMatlab is deprecated and will be removed in a future version. "
            "Use direct MATLAB expression evaluation or LaTeX parsing instead.",
            DeprecationWarning,
            stacklevel=2
        )
    
    def sympy_to_matlab(self, expr):
        """
        DEPRECATED: Convert SymPy expression to MATLAB format.
        This method is deprecated and will be removed in a future version.
        """
        warnings.warn(
            "sympy_to_matlab is deprecated and will be removed in a future version. "
            "Use direct MATLAB expression evaluation or LaTeX parsing instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return str(expr)
