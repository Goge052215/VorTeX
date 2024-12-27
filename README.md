# MATLAB-Scientific Calculator on Py

![legend img](legend_img.png)

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

## Setup Instructions

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Goge052215/MATLAB-Calculator-on-py.git
   ```

2. **Install Dependencies:**

   Ensure you have MATLAB installed and the MATLAB Engine API for Python set up.

   ```bash
   pip install -r requirements.txt
   ```

   The requirement list is in [requirements.txt](requirements.txt)

3. **Run the Application:**

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
  - **Differentiation:** `d/dx(x^2)`, `d2/dx2(x^2)`, etc.
  - **Integration:** `int e^(x) dx`, `int ln(x) dx`, `int x^2 dx`, etc.
  - **Trigonometric:** `sin(30)`, `cos(30)`, `tan(30)`, etc.

- **MATLAB Mode:**
  - **Differentiation:** `diff(x^2, x)`, `diff(x^2, x, 2)`, etc.
  - **Integration:** `int(exp(x), x)`, `int(ln(x), x)`, `int(x^2, x)`, etc.
  - **Trigonometric:** `sin(30)`, `cos(30)`, `tan(30)`, etc.

- **Matrix Mode:**
  - **Determinant:** `[1 2; 3 4]`
  - **Inverse:** `[1 0; 0 1]`

*Note:* Simplified $\LaTeX$ input is recommended, for a guide of simplified $\LaTeX$ input, see the table below:

| $\LaTeX$ Input | Simplified $\LaTeX$ Input |
| ------------- | ----------------------- |
| `\frac{d}{dx}(x^2)` |   `d/dx (x^2)`    |
| `\int e^{x} dx`| `int e^x dx`           |
| `\int_{a}^{b} f(x) dx` | `int a to b f(x) dx` |
| `\frac{d^2}{dx^2}(x^2)` | `d2/dx2 (x^2)` (works for rest of higher order derivatives) |
| `\sin{30}`     | `sin(30)` (works for rest of trigonometric functions)             |
| `\sqrt{x}`     |      `sqrt(x)`         |
| `\abs{x}`      |       `abs(x)`         |
| `\ln(x)`       | `ln(x)`                |
| `\log_{10}(x)` | `log10(x)`             |
| `\log_{n}(x)`  | `logn(x)`              |
| `\alpha`       | `alpha` (works for rest of greek letters)            |
| `\sum`         | `sum`                  |
| `\prod`        | `prod`                 |
| `\lim`         | `lim`                  |
| `\infty`       | `infty`                |

for more shortcuts, see [shortcut.py](latex_pack/shortcut.py)

### Example for $\LaTeX$ Mode

```LaTeX
int 1/x dx           ->  ln(x)
d2/dx2 (4x^10)       ->  320x^8
tan(90) or tan(pi/2) ->  inf
```

## Troubleshooting

- **MATLAB Engine Not Starting:**
  - Ensure that MATLAB is installed on your system.
  - Verify that the MATLAB Engine API for Python is correctly installed.
  - Check environment variables and MATLAB's path settings.

- **Invalid Expression Errors:**
  - Ensure that your $\LaTeX$ or MATLAB expressions are correctly formatted.
  - Verify that all necessary functions are supported and properly replaced.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

[MIT License](LICENSE)
