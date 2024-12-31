# MATLAB-Scientific Calculator on Py

![legend img](imgs/legend_img.png)

A PyQt5-based scientific calculator that supports $\LaTeX$ input, integrates with MATLAB for symbolic computation, and offers various mathematical functionalities including differentiation, integration, and matrix operations.

## Features

- **$\LaTeX$ Input:** Enter mathematical expressions in $\LaTeX$ format for easy readability and input.
- **MATLAB Integration:** Utilize MATLAB's symbolic toolbox for advanced computations, ensuring high precision and reliability.
- **Trigonometric Functions:** Supports both Degree and Radian modes for trigonometric calculations.
- **Symbolic Computation:** Handle derivatives and integrals symbolically, providing exact results where possible.
- **Matrix Operations:** Perform operations such as determinant, inverse, eigenvalues, and more on matrices.
- **Theming:** Multiple UI themes available to customize the appearance of the application.
- **Logging:** Detailed logging of operations and errors for troubleshooting and analysis.
- **Auto Simplify:** Automatically simplifies the result of the expression.

## Getting Started

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Goge052215/MATLAB-Calculator-on-py.git
   ```

2. **Install Required Fonts**

   Thanks for developers of Monaspace font family! We are using the font [**Monaspace Neon**](https://monaspace.githubnext.com/) for the UI (partially).You can install the font by running the scripts in [fonts](fonts) folder.

   So far, there are 2 scripts for following systems:
   - [Windows](fonts/fonts_download.ps1)
   - [MacOS](fonts/fonts_download.bash)

   For Linux users, please see the instructions in the [GitHub Monaspace repository](https://github.com/githubnext/monaspace?tab=readme-ov-file).

3. **Install Dependencies:**

   Ensure you have MATLAB installed and the MATLAB Engine API for Python set up.

   ```bash
   pip install -r requirements.txt
   ```

   The requirement list is in [requirements.txt](requirements.txt)

4. **Run the Application:**

   ```bash
   python main.py
   ```

## Usage

1. **Select Input Mode:**
   - **$\LaTeX$:** Enter expressions in $\LaTeX$ format for symbolic computation.
   - **MATLAB:** Enter raw MATLAB expressions for direct evaluation.
   - **Matrix:** Perform matrix operations (e.g., determinant, inverse).

2. **Select Angle Mode:**
   - **Degree:** Use degree mode for trigonometric functions.
   - **Radian:** Use radian mode for trigonometric functions.

3. **Enter Expression:**
   - In the input field, type your mathematical expression.

4. **Calculate:**
   - Click the "Calculate" button to evaluate the expression.

5. **View Result:**
   - The result will be displayed below the calculate button.

## Example Expressions

- **Simplified $\LaTeX$ Mode:**
  - **Differentiation:** `d/dx (x^2)`, `d2/dx2 (x^2)`, etc.
  - **Integration:** `int e^(x) dx`, `int ln(x) dx`, `int x^2 dx`, etc.
  - **Trigonometric:** `sin(30)`, `cos(30)`, `tan(30)`, etc.

- **MATLAB Mode:**
  - **Differentiation:** `diff(x^2, x)`, `diff(x^2, x, 2)`, etc.
  - **Integration:** `int(exp(x), x)`, `int(ln(x), x)`, `int(x^2, x)`, etc.
  - **Trigonometric:** `sin(30)`, `cos(30)`, `tan(30)`, etc.

*Note:* Simplified $\LaTeX$ input is recommended, for a guide of simplified $\LaTeX$ input, see the table below:

| $\LaTeX$ | Simplified $\LaTeX$ Input |
| ------------- | ----------------------- |
| $\displaystyle\frac{\text{d}}{\text{d}x}(f(x))$ |  `d/dx (f(x))`  |
| $\displaystyle\frac{\text{d}^n}{\text{d}x^n}(f(x))$  | `dn/dxn (f(x))` |
| $\displaystyle\int e^{x} dx$ | `int e^x dx`           |
| $\displaystyle\int_{a}^{b} f(x) dx$ | `int a to b f(x) dx` |
| $\sin, \cos, \tan, \dots$ | `sin, cos, tan, ...` |
| $\displaystyle\binom{n}{r}$ or $^n\text{C}_r$ | `binom(n, r)` |
| $\sqrt{x}$ | `sqrt(x)` |
| $\|x\|$  | `abs(x)`  |
| $\ln(x)$   | `ln(x)`   |
| $\log_{10}(x)$ | `log10(x)` |
| $\log_{n}(x)$  | `logn(x)`  |
| $\alpha, \beta, \gamma, \dots$ | `alpha, beta, gamma, ...` |
| $\sum$         | `sum`      |
| $\prod$        | `prod`     |
| $\lim$         | `lim`      |
| $\infty$       | `infty`    |

for more shortcuts, see [shortcut.py](latex_pack/shortcut.py)

### Example for $\LaTeX$ Mode

1. `int 1/x dx` $\rightarrow$ $\displaystyle \int \frac{1}{x} \, \text{d}x = \ln(x)$
2. `d2/dx2 (4x^10)` $\rightarrow$ $\displaystyle \frac{\text{d}^2}{\text{d}x^2} (4x^{10}) = 320x^8$
3. `binom(5, 2)` $\rightarrow$ $\displaystyle \binom{5}{2} = 10$
4. `tan(90) or tan(pi/2)` $\rightarrow$ $\tan(90) = \infty$

### Example for Matrix Mode

1. (Determinant Mode) `[1 2; 3 4]` $\rightarrow$ $\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} = -2$
2. (Inverse Mode) `[1 2; 3 4]` $\rightarrow$ $\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}^{-1} = \begin{pmatrix} -2 & 1 \\ 1.5 & -0.5 \end{pmatrix} \rightarrow$ `[[-2.0, 1.0], [1.5, -0.5]]`
3. (Eigenvalues Mode) `[1 2; 3 4]` $\rightarrow$ $\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} = \begin{pmatrix} -0.37 & 0.00 \\ 0.00 & 5.37 \end{pmatrix} \rightarrow$ `[-0.37, 5.37]`
4. (Rank Mode) `[1 2; 3 4]` $\rightarrow$ $\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} = 2$

## Troubleshooting

- **MATLAB Engine Not Starting:**
  - Ensure that MATLAB is installed on your system.
  - Verify that the MATLAB Engine API for Python is correctly installed.
  - Check environment variables and MATLAB's path settings.

- **Invalid Expression Errors:**
  - Ensure that your $\LaTeX$ or MATLAB expressions are correctly formatted.
  - Verify that all necessary functions are supported and properly replaced.

### Current TODO

- [ ] Fixing the limits handling
- [ ] Fixing the expression handling for $^n\text{C}_r$
- [ ] Fixing the series evaluation

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

[MIT License](LICENSE)
