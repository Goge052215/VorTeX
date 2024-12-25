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
        self.assertEqual(matlab_expr, "2*x + 3")

    def test_logarithms(self):
        expr_ln = sy.log(2)  # Natural log
        matlab_expr_ln = self.converter.sympy_to_matlab(expr_ln)
        self.assertEqual(matlab_expr_ln, "log(2)")

        expr_log10 = sy.log(2, 10)  # Base-10 log
        matlab_expr_log10 = self.converter.sympy_to_matlab(expr_log10)
        self.assertEqual(matlab_expr_log10, "log(2)/log(10)")

    def test_log_with_other_base(self):
        expr_log3 = sy.log(2, 3)
        matlab_expr_log3 = self.converter.sympy_to_matlab(expr_log3)
        self.assertEqual(matlab_expr_log3, "log(2)/log(3)")

    def test_third_derivative(self):
        expr = sy.Derivative(sy.cos(sy.Symbol('x')), (sy.Symbol('x'), 3))
        matlab_expr = self.converter.sympy_to_matlab(expr)
        self.assertEqual(matlab_expr, "diff(cos(x), 'x', 3)")

    def test_mixed_expression(self):
        expr = sy.Derivative(sy.Integral(sy.exp(sy.Symbol('x')), sy.Symbol('x')), 'x')
        matlab_expr = self.converter.sympy_to_matlab(expr)
        self.assertEqual(matlab_expr, "diff(int(exp(x), 'x'), 'x')")

    def test_multiple_integrals(self):
        expr = sy.Integral(sy.Integral(sy.sin(sy.Symbol('x')), sy.Symbol('x')), sy.Symbol('x'))
        matlab_expr = self.converter.sympy_to_matlab(expr)
        self.assertEqual(matlab_expr, "int(int(sin(x), 'x'), 'x')")

    def test_partial_derivative(self):
        x, y = sy.symbols('x y')
        expr = sy.Derivative(sy.sin(x*y), x)
        matlab_expr = self.converter.sympy_to_matlab(expr)
        self.assertEqual(matlab_expr, "diff(sin(x*y), 'x')")

if __name__ == '__main__':
    unittest.main()
