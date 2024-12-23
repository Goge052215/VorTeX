# Scientific Calculator

A PyQt5-based scientific calculator that supports LaTeX input, integrates with MATLAB for symbolic computation, and offers various mathematical functionalities including differentiation and integration.

## Features

- **LaTeX Input:** Enter mathematical expressions in LaTeX format.
- **MATLAB Integration:** Utilize MATLAB's symbolic toolbox for advanced computations.
- **Trigonometric Functions:** Supports both Degree and Radian modes.
- **Symbolic Computation:** Handle derivatives and integrals.
- **Theming:** Multiple UI themes available.

## Setup Instructions

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/scientific_calculator.git
   cd scientific_calculator
   ```

2. **Create a Virtual Environment (Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   Ensure you have MATLAB installed and the MATLAB Engine API for Python set up.

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application:**

   ```bash
   python main.py
   ```

## Usage

1. **Select Input Mode:**
   - **LaTeX:** Enter expressions in LaTeX format.
   - **MATLAB:** Enter raw MATLAB expressions.
   - **Matrix:** [To be implemented]

2. **Select Angle Mode:**
   - **Degree**
   - **Radian**

3. **Enter Expression:**
   - In the input field, type your mathematical expression.

4. **Calculate:**
   - Click the "Calculate" button to evaluate the expression.

5. **View Result:**
   - The result will be displayed below the calculate button.

## Example Expressions

- **LaTeX Mode:**
  - **Differentiation:** `\frac{d}{dx}(x^2)`
  - **Integration:** `\int{e^{x}} dx`
  - **Trigonometric:** `\sin{30}`

- **MATLAB Mode:**
  - **Differentiation:** `diff(x^2, x)`
  - **Integration:** `int(exp(x), x)`
  - **Trigonometric:** `sin(30)`

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
