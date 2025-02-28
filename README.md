<div align="center">

# ⚡ VorTeX ⚡

> *Where Mathematics Converges*

Are you tired of: \
Juggling between multiple math applications? 📱 \
Wrestling with complex $\LaTeX$ syntax? ⚔️ \
Hitting the limitations of basic calculators? 🚫 \
Running MATLAB for simple calculations? 💻

**VorTeX** seamlessly fuses them all into one powerful solution.

![legend img](imgs/legend_img.png)

### A Mathematical Vortex of Power

[![MATLAB](https://img.shields.io/badge/MATLAB-Powered-blue.svg)](https://www.mathworks.com/products/matlab.html)
[![LaTeX](https://img.shields.io/badge/LaTeX-Simplified-green.svg)](https://www.latex-project.org/)
[![PyQt5](https://img.shields.io/badge/PyQt5-Modern_UI-orange.svg)](https://www.riverbankcomputing.com/software/pyqt/)
[![Manim](https://img.shields.io/badge/Manim-Visualizations-red.svg)](https://www.manim.community/)

---

A powerful mathematical workbench that fuses **MATLAB** and **SymPy**'s symbolic computation, **LaTeX**'s elegant notation, **PyQt5**'s modern interface, and **Manim**'s visualization capabilities into one seamless vortex. *VorTeX* brings together advanced mathematical tools for differentiation, integration, matrix operations, and symbolic computation in an intuitive environment.

---

</div>

## Features

- **$\LaTeX$ Input:** Enter mathematical expressions in $\LaTeX$ format for easy readability and input.
- **MATLAB Integration:** Utilize MATLAB's symbolic toolbox for advanced computations, ensuring high precision and reliability.
- **Trigonometric Functions:** Supports both Degree and Radian modes for trigonometric calculations.
- **Symbolic Computation:** Handle derivatives and integrals symbolically, providing exact results where possible.
- **Matrix Operations:** Perform operations such as determinant, inverse, eigenvalues, and more on matrices.
- **SymPy Integration:** Utilize SymPy's symbolic computation for non-MATLAB users.
- **Theming:** Multiple UI themes available to customize the appearance of the application.
- **Logging:** Detailed logging of operations and errors for troubleshooting and analysis.
- **Auto Simplify:** Automatically simplifies the result of the expression.
- **Manim Visualization:** Visualize the expression using Manim. (Currently only limited function is supported)

## Getting Started

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Goge052215/VorTeX.git
   ```

2. **Install Required Fonts**

   Thanks for developers of Monaspace font family! We are using the font [**Monaspace Neon**](https://monaspace.githubnext.com/) for the UI (partially).You can install the font by running the scripts in [fonts](fonts) folder.

   The system will go through an [iP check](ip_check/ip_check.py) to see whether you can access GitHub easily, then we will check the font and automatically download for you :)

   If automatic download does not work, try the manual way! So far, there are 2 scripts for following systems:
   - [Windows](fonts/fonts_download.ps1)
   - [MacOS](fonts/fonts_download.bash)

   For Linux and other system users, please see the instructions in the [GitHub Monaspace repository](https://github.com/githubnext/monaspace?tab=readme-ov-file).

4. **Install Dependencies:**

   Ensure you have MATLAB installed and the MATLAB Engine API for Python set up. Please ensure you put your MATLAB app (e.g. MATLABr2024b.app) address in PATH!

   ```bash
   pip install -r requirements.txt
   ```

   For non-MATLAB users, you can use SymPy instead, we also provided SymPy calculation module for this calculator.

   The requirement list is in [requirements.txt](requirements.txt)

5. **Run the Application:**

   ```bash
   python main.py
   ```

## Usage

1. **Select Input Mode:**
   - **$\LaTeX$:** Enter expressions in $\LaTeX$ format for symbolic computation.
   - **MATLAB:** Enter raw MATLAB expressions for direct evaluation.
   - **Matrix:** Perform matrix operations (e.g., determinant, inverse).
   - **SymPy:** Enter Simplified $\LaTeX$ expressions (for non-MATLAB users).

2. **Select Angle Mode:**
   - **Degree:** Use degree mode for trigonometric functions.
   - **Radian:** Use radian mode for trigonometric functions.

3. **Enter Expression:**
   - In the input field, type your mathematical expression.

4. **Calculate:**
   - Click the "Calculate" button to evaluate the expression.

5. **View Result:**
   - The result will be displayed below the calculate button.

6. **Visualize:**
   - Click the "Visualize" button to visualize the expression.
   - The expression will be processed under Manim module and displayed in .mp4 file

7. **Settings:**
   - Click the "Settings" button to open the settings window.
   - The settings window allows you to customize the application's behavior and appearance, also you can check the debug logs here.

## Example Expressions

- **Simplified $\LaTeX$ Mode, SymPy Mode:**
  - **Differentiation:** `d/dx (x^2)`, `d2/dx2 (x^2)`, etc.
  - **Integration:** `int e^(x) dx`, `int ln(x) dx`, `int x^2 dx`, etc.
  - **Trigonometric:** `sin(30)`, `cos(30)`, `tan(30)`, etc.

- **MATLAB Mode:**
  - **Differentiation:** `diff(x^2, x)`, `diff(x^2, x, 2)`, etc.
  - **Integration:** `int(exp(x), x)`, `int(ln(x), x)`, `int(x^2, x)`, etc.
  - **Trigonometric:** `sin(30)`, `cos(30)`, `tan(30)`, etc.

*Note:* Simplified $\LaTeX$ input is recommended, for a guide of simplified $\LaTeX$ input, see the table below:

| $\LaTeX$ | Previous $\LaTeX$ Command | Simplified $\LaTeX$ Input |
| ------------- | ----------------------- | ----------------------- |
| $\displaystyle\frac{\text{d} }{\text{d}x}(f(x))$ | `\frac{d}{dx} (f(x))` | `d/dx (f(x))`  |
| $\displaystyle\frac{\text{d}^n}{\text{d}x^n}(f(x))$  | `\frac{d^n}{dx^n} (f(x))` | `dn/dxn (f(x))` |
| $\displaystyle\int e^{x} \text{d}x$ | `\int e^{x} dx` | `int e^x dx` |
| $\displaystyle\int_{a}^{b} f(x) \text{d}x$ | `\int_{a}^{b} f(x) dx` | `int (a to b) f(x) dx` |
| $\sin, \cos, \tan, \dots$ | `\sin, \cos, \tan, ...` | `sin, cos, tan, ...` |
| $\displaystyle\binom{n}{r}$ or $^n\text{C}_r$ | `\binom{n}{r}` | `binom(n, r) or nCr` |
| $\sqrt{x}$  | `\sqrt{x}` | `sqrt(x)` |
| $\|x\|$  | `\|x\|` | `abs(x)` |
| $\ln(x)$   | `\ln(x)` | `ln(x)` |
| $\log_{10}(x)$ | `\log_{10}(x)` | `log10(x)` |
| $\log_{n}(x)$  | `\log_{n}(x)`  | `logn(x)`  |
| $\alpha, \beta, \gamma, \dots$ | `\alpha, \beta, \gamma, ...` | `alpha, beta, gamma, ...` |
| $\displaystyle\sum_{i = a}^{b} f(x_i)$         | `\sum_{i = a}^{b} f(x_i)` | `sum (a to b) f(x)`      |
| $\displaystyle\prod_{i = a}^{b} f(x_i)$      | `\prod_{i = a}^{b} f(x_i)` | `prod (a to b) f(x)`     |
| $\displaystyle\lim_{x \to a} f(x)$         | `\lim_{x \to a} f(x)` | `lim (x to a) f(x)`      |
| $\displaystyle\lim_{x \to a^+} f(x)$       | `\lim_{x \to a^+} f(x)` | `lim (x to a+) f(x)`      |
| $\displaystyle\lim_{x \to a^-} f(x)$       | `\lim_{x \to a^-} f(x)` | `lim (x to a-) f(x)`      |
| $\pm\infty$       | `\pm\infty`    | `+infty` or `-infty`   |

for more shortcuts, see [shortcut.py](latex_pack/shortcut.py)

### Example for $\LaTeX$ Mode, SymPy Mode

1. `int 1/x dx` $\rightarrow$ $\displaystyle \int \frac{1}{x} \text{d}x = \ln(x)$
2. `int (1 to 3) x^3/(x^2 + 1) dx` $\rightarrow$ $\displaystyle \int_{1}^{3} \frac{x^3}{x^2 + 1} \text{d}x = 4 - \left(\frac{\ln 5}{2}\right)$
3. `d2/dx2 (4x^10)` $\rightarrow$ $\displaystyle \frac{\text{d}^2}{\text{d}x^2} (4x^{10}) = 360x^8$
4. `binom(5, 2) or 5C2` $\rightarrow$ $\displaystyle \binom{5}{2} = 10$
5. `tan(90) or tan(pi/2)` $\rightarrow$ $\tan(90) = \infty$
6. `sum (1 to 100) x` $\rightarrow$ $\displaystyle \sum_{i = 1}^{100} x = 5050$
7. `prod (2 to 10) ln(x)` $\rightarrow$ $\displaystyle \prod_{i = 2}^{10} \ln(x) = 62.321650$
8. `lim (x to 0) sin(x)/x` $\rightarrow$ $\displaystyle \lim_{x \to 0} \frac{\sin(x)}{x} = 1$

### Example for Matrix Mode

1. (Determinant Mode) `[1 2; 3 4]`
```math
\begin{bmatrix}
   1 & 2 \\
   3 & 4
\end{bmatrix} = -2
```

2. (Inverse Mode) `[1 2; 3 4]`
```math
\begin{pmatrix}
   1 & 2 \\
   3 & 4
\end{pmatrix}^{-1} = \begin{pmatrix}
   -2 & 1 \\
   1.5 & -0.5
\end{pmatrix}
```
Output: `[[-2.0, 1.0], [1.5, -0.5]]`

3. (Eigenvalues Mode) `[1 2; 3 4]`
```math
\begin{pmatrix}
   1 & 2 \\
   3 & 4
\end{pmatrix} \rightarrow \begin{pmatrix}
   -0.37 & 0.00 \\
   0.00 & 5.37
\end{pmatrix}
```
Output: `[-0.37, 5.37]`

4. (Rank Mode) `[1 2; 3 4]`
```math
\begin{pmatrix}
   1 & 2 \\
   3 & 4
\end{pmatrix} = 2
```

## Troubleshooting

- **MATLAB Engine Not Starting:**
  - Ensure that MATLAB is installed on your system.
  - Verify that the MATLAB Engine API for Python is correctly installed.
  - Check environment variables and MATLAB's path settings.

- **Invalid Expression Errors:**
  - Ensure that your Simplified $\LaTeX$ or MATLAB expressions are correctly formatted.
  - Verify that all necessary functions are supported and properly replaced.

- More issues will be added in the future debugging.

### Current TODO

- [X] Fixing the limits handling
- [X] Fixing the expression handling for $^n\text{C}_r$
- [X] Fixing the series evaluation
- [X] Adding 2D plot Manim for functions
- [ ] 2D plot Manim for calculus
- [ ] Adding 3D demonstration Manim

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

[MIT License](LICENSE)
