"""
Legend text module for displaying LaTeX and Matrix command legends.
Provides formatted HTML tables for visual reference.
"""

SIMPLE_LATEX_LEGEND = """
<table style="color: #a9b1d6; border-collapse: collapse; width: 100%;">
    <tr>
        <th style="text-align: left; padding: 4px; border: 1px solid #414868;">Operation</th>
        <th style="text-align: left; padding: 4px; border: 1px solid #414868;">LaTeX Command</th>
    </tr>
    <tr>
        <td style="padding: 4px; border: 1px solid #414868;">Combinations (nCr)</td>
        <td style="padding: 4px; border: 1px solid #414868;"><code>binom(n,r)</code></td>
    </tr>
    <tr>
        <td style="padding: 4px; border: 1px solid #414868;">Trigonometric Functions</td>
        <td style="padding: 4px; border: 1px solid #414868;"><code>sin, cos, tan</code></td>
    </tr>
    <tr>
        <td style="padding: 4px; border: 1px solid #414868;">Exponentiation</td>
        <td style="padding: 4px; border: 1px solid #414868;"><code>x^2</code> (e.g., <code>2^3</code>)</td>
    </tr>
    <tr>
        <td style="padding: 4px; border: 1px solid #414868;">Factorial</td>
        <td style="padding: 4px; border: 1px solid #414868;"><code>5!</code></td>
    </tr>
    <tr>
        <td style="padding: 4px; border: 1px solid #414868;">Square Root</td>
        <td style="padding: 4px; border: 1px solid #414868;"><code>sqrt{16}</code></td>
    </tr>
    <tr>
        <td style="padding: 4px; border: 1px solid #414868;">Logarithm</td>
        <td style="padding: 4px; border: 1px solid #414868;"><code>log10, ln, logn</code> (natural log)</td>
    </tr>
    <tr>
        <td style="padding: 4px; border: 1px solid #414868;">Constants</td>
        <td style="padding: 4px; border: 1px solid #414868;"><code>pi</code>, <code>e</code></td>
    </tr>
    <tr>
        <td style="padding: 4px; border: 1px solid #414868;">Absolute Value</td>
        <td style="padding: 4px; border: 1px solid #414868;"><code>abs</code></td>
    </tr>
    <tr>
        <td style="padding: 4px; border: 1px solid #414868;">Fraction</td>
        <td style="padding: 4px; border: 1px solid #414868;"><code>a/b,\\frac{a}{b}</code></td>
    </tr>
    <tr>
        <td style="padding: 4px; border: 1px solid #414868;">Definite Integral</td>
        <td style="padding: 4px; border: 1px solid #414868;"><code>int (a to b) f(x) dx</code></td>
    </tr>
    <tr>
        <td style="padding: 4px; border: 1px solid #414868;">Indefinite Integral</td>
        <td style="padding: 4px; border: 1px solid #414868;"><code>int f(x) dx</code></td>
    </tr>
    <tr>
        <td style="padding: 4px; border: 1px solid #414868;">Derivative</td>
        <td style="padding: 4px; border: 1px solid #414868;"><code>d/dx, d2/dx2, d3/dx3</code></td>
    </tr>
    <tr>
        <td style="padding: 4px; border: 1px solid #414868;">Sum</td>
        <td style="padding: 4px; border: 1px solid #414868;"><code>sum (a to b)</code></td>
    </tr>
    <tr>
        <td style="padding: 4px; border: 1px solid #414868;">Products</td>
        <td style="padding: 4px; border: 1px solid #414868;"><code>prod (a to b)</code></td>
    </tr>
    <tr>
        <td style="padding: 4px; border: 1px solid #414868;">Limits</td>
        <td style="padding: 4px; border: 1px solid #414868;"><code>lim (x to a)</code></td>
    </tr>
</table>
"""

MATRIX_LEGEND = """
<table style="color: #a9b1d6; border-collapse: collapse; width: 100%;">
    <tr>
        <th style="text-align: left; padding: 4px; border: 1px solid #414868;">Matrix Input</th>
        <th style="text-align: left; padding: 4px; border: 1px solid #414868;">Example</th>
    </tr>
    <tr>
        <td style="padding: 4px; border: 1px solid #414868;">2x2 Matrix</td>
        <td style="padding: 4px; border: 1px solid #414868;"><code>[1 2; 3 4]</code></td>
    </tr>
    <tr>
        <td style="padding: 4px; border: 1px solid #414868;">3x3 Matrix</td>
        <td style="padding: 4px; border: 1px solid #414868;"><code>[1 2 3; 4 5 6; 7 8 9]</code></td>
    </tr>
    <tr>
        <td style="padding: 4px; border: 1px solid #414868;">Column Vector</td>
        <td style="padding: 4px; border: 1px solid #414868;"><code>[1; 2; 3]</code></td>
    </tr>
    <tr>
        <td style="padding: 4px; border: 1px solid #414868;">Row Vector</td>
        <td style="padding: 4px; border: 1px solid #414868;"><code>[1 2 3]</code></td>
    </tr>
</table>
"""