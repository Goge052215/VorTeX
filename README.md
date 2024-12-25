# MATLAB-Scientific Calculator on Py

A PyQt5-based scientific calculator that supports LaTeX input, integrates with MATLAB for symbolic computation, and offers various mathematical functionalities including differentiation, integration, and matrix operations.

## Features

- **LaTeX Input:** Enter mathematical expressions in LaTeX format for easy readability and input.
- **MATLAB Integration:** Utilize MATLAB's symbolic toolbox for advanced computations, ensuring high precision and reliability.
- **Trigonometric Functions:** Supports both Degree and Radian modes for trigonometric calculations.
- **Symbolic Computation:** Handle derivatives and integrals symbolically, providing exact results where possible.
- **Matrix Operations:** Perform operations such as determinant, inverse, eigenvalues, and more on matrices.
- **Theming:** Multiple UI themes available to customize the appearance of the application.
- **Logging:** Detailed logging of operations and errors for troubleshooting and analysis.

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

3. **Run the Application:**

   ```bash
   python main.py
   ```

## Usage

1. **Select Input Mode:**
   - **LaTeX:** Enter expressions in LaTeX format for symbolic computation.
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

- **LaTeX Mode:**
  - **Differentiation:** `d/dx(x^2)`
  - **Integration:** `int e^(x) dx`, `int ln(x) dx`, `int x^2 dx`
  - **Trigonometric:** `sin{30}`

- **MATLAB Mode:**
  - **Differentiation:** `diff(x^2, x)`
  - **Integration:** `int(exp(x), x)`
  - **Trigonometric:** `sin(30)`

- **Matrix Mode:**
  - **Determinant:** `[1 2; 3 4]`
  - **Inverse:** `[1 0; 0 1]`

## Troubleshooting

- **MATLAB Engine Not Starting:**
  - Ensure that MATLAB is installed on your system.
  - Verify that the MATLAB Engine API for Python is correctly installed.
  - Check environment variables and MATLAB's path settings.

- **Invalid Expression Errors:**
  - Ensure that your LaTeX or MATLAB expressions are correctly formatted.
  - Verify that all necessary functions are supported and properly replaced.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

[MIT License](LICENSE)
