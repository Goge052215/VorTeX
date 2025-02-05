import unittest
from parameterized import parameterized
import sympy as sy
import sys
import os
from sympy import Integral, Derivative, Symbol, sin, cos, exp, log

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matlab_interface.sympy_to_matlab import SympyToMatlab

class TestSympyToMatlabConversion(unittest.TestCase):
    """Modernized test suite for Sympy-to-MATLAB conversion"""

    @classmethod
    def setUpClass(cls):
        cls.converter = SympyToMatlab()
        cls.x = Symbol('x')
        cls.y = Symbol('y')

    # Region: Parameterized Core Tests
    @parameterized.expand([
        ("basic_arithmetic", '2*x + 3', "2*x + 3"),
        ("indefinite_integral", Integral(1/Symbol('x'), Symbol('x')), "int(1/x, 'x')"),
        ("definite_integral", Integral(1/Symbol('x'), (Symbol('x'), 1, 2)), 
         "int(1/x, 'x', 1, 2)"),
        ("first_derivative", Derivative(sin(Symbol('x')), Symbol('x')), 
         "diff(sin(x), 'x')"),
        ("second_derivative", Derivative(sin(Symbol('x')), (Symbol('x'), 2)), 
         "diff(sin(x), 'x', 2)"),
        ("natural_log", log(2), "log(2)"),
        ("log_base_10", log(2, 10), "log(2)/log(10)"),
        ("log_base_3", log(2, 3), "log(2)/log(3)"),
        ("third_derivative", Derivative(cos(Symbol('x')), (Symbol('x'), 3)), 
         "diff(cos(x), 'x', 3)"),
        ("mixed_expression", Derivative(Integral(exp(Symbol('x')), Symbol('x')), Symbol('x')), 
         "diff(int(exp(x), 'x'), 'x')"),
        ("double_integral", Integral(Integral(sin(Symbol('x')), Symbol('x')), Symbol('x')), 
         "int(int(sin(x), 'x'), 'x')"),
        ("partial_derivative", Derivative(sin(Symbol('x')*Symbol('y')), Symbol('x')), 
         "diff(sin(x*y), 'x')"),
    ])
    def test_conversion_cases(self, name, expr, expected):
        """Parameterized test covering all core conversion scenarios"""
        result = self.converter.sympy_to_matlab(expr)
        self.assertEqual(result, expected)

    # Region: Special Cases
    def test_matrix_expression(self):
        matrix = sy.Matrix([[self.x**2, 1/self.x], [0, 1]])
        expected = "[x^2, 1/x; 0, 1]"
        self.assertEqual(self.converter.sympy_to_matlab(matrix), expected)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            self.converter.sympy_to_matlab("not_a_sympy_object")

    def test_symbolic_constants(self):
        expr = sy.exp(sy.pi)
        self.assertEqual(self.converter.sympy_to_matlab(expr), "exp(pi)")

if __name__ == '__main__':
    unittest.main(failfast=True, verbosity=2)
