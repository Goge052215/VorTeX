import unittest
import sympy as sy
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matlab_interface.sympy_to_matlab import SympyToMatlab

class TestSympyToMatlab(unittest.TestCase):
    def setUp(self):
        self.converter = SympyToMatlab()

    def test_indefinite_integral(self):
        expr = sy.Integral(1/sy.Symbol('x'), sy.Symbol('x'))
        matlab_expr = self.converter.sympy_to_matlab(expr)
        self.assertEqual(matlab_expr, "int(1/x, 'x')")

    def test_definite_integral(self):
        expr = sy.Integral(1/sy.Symbol('x'), (sy.Symbol('x'), 1, 2))
        matlab_expr = self.converter.sympy_to_matlab(expr)
        self.assertEqual(matlab_expr, "int(1/x, 'x', 1, 2)")

    def test_first_derivative(self):
        expr = sy.Derivative(sy.sin(sy.Symbol('x')), 'x')
        matlab_expr = self.converter.sympy_to_matlab(expr)
        self.assertEqual(matlab_expr, "diff(sin(x), 'x')")

    def test_second_derivative(self):
        expr = sy.Derivative(sy.sin(sy.Symbol('x')), (sy.Symbol('x'), 2))
        matlab_expr = self.converter.sympy_to_matlab(expr)
        self.assertEqual(matlab_expr, "diff(sin(x), 'x', 2)")

    def test_basic_expression(self):
        expr = sy.sympify('2*x + 3')
        matlab_expr = self.converter.sympy_to_matlab(expr)
        print(f"matlab_expr: {matlab_expr} (type: {type(matlab_expr)})")  # Debugging line
        self.assertEqual(matlab_expr, "2*x + 3")

    def test_numeric_expression(self):
        expr = sy.sin(sy.Symbol('x'))
        matlab_expr = self.converter.sympy_to_matlab(expr)
        self.assertEqual(matlab_expr, "sind(x)")  # Assuming degrees

        # Further testing would require MATLAB integration to check actual numeric outputs

if __name__ == '__main__':
    unittest.main()
