### 1. Basic Arithmetic Operations

| Test Case No | Math Expression | Expected Output | STATUS |
| ------------ | --------------- | --------------- | ------ |
| `BA1`        | $2+3 \times 4$  | $14$            | `PASS` |
| `BA2`        | $2^{10}$        | $1024$          | `PASS` |
| `BA3`   |$\text{sqrt}(16)+\text{cbrt}(27)$| $7$  | `PASS` |
| `BA4`       |$(x+1)^2 - (x-1)^2$| $4x$           | `PASS` |

### 2. Trigonometric Functions (Degree Mode)

| Test Case No | Math Expression | Expected Output |      STATUS      |
| ------------ | --------------- | --------------- | ---------------- |
|`TF1`|$\displaystyle\sin\left(\frac{\pi}{2}\right)$| $0.5$ | `PASS`  |
|`TF2`|$\displaystyle\cos\left(\pi\right)$          | $-1$  | `PASS`  |
|`TF3`|$\displaystyle\tan\left(\frac{\pi}{4}\right)$| $1$   | `PASS`  |
|`TF4`|$\displaystyle\left(\sin\left(\frac{\pi}{6}\right)\right)^{2}+\left(\cos\left(\frac{\pi}{6}\right)\right)^{2}$| $1$ | `PASS`  |

### 3. Logarithmic and Exponential Functions

| Test Case No | Math Expression | Expected Output |      STATUS      |
| ------------ | --------------- | --------------- | ---------------- |
|`LF1`         |$e^{1}$          | $e$             | `PASS`           |
|`LF2`         |$\ln\left(e^{3}\right)$| $3$       | `PASS`           |
|`LF3`         |$\log_{10}(1000)$| $3$             | `PASS`           |
|`LF4`         |$\log_{3}(9)$    | $2$             | `PASS`           |
|`LF5`         |$\log_{2}(8)$    | $3$             | `PASS`           |
|`LF6`         |$\log_{15}(125)$ | $1.12$          | `PASS`           |
|`LF7`         |$\log_{7}(49)$   | $2$             | `PASS`           |
|`LF8`         |$\log_{5}(\sqrt{5})$| $0.5$        | `PASS`           |

### 4. Calculus

| Test Case No | Math Expression | Expected Output |             STATUS              |
| ------------ | --------------- | --------------- | ------------------------------- |
|`CAL1`|$\displaystyle\frac{\text{d}}{\text{d}x}\left(x^{2}\right)$| $2x$ | `PASS`   |
|`CAL2`|$\displaystyle\int x^2 \, \text{d}x$|$\displaystyle\frac{x^{3}}{3} + C$|`PASS` |
|`CAL3`|$\displaystyle\int_{0}^{1} x^2 \, \text{d}x$|$\displaystyle \frac{1}{3}$|`PASS`|
|`CAL4`|$\displaystyle\lim_{x\to 0} \frac{\sin(x)}{x}$| $1$               |  `PASS`  |
|`CAL5`|$\displaystyle\int_{-\infty}^{\infty} e^{-x^2} \,\text{d}x$|$\sqrt{x}$|`PASS`|
|`CAL6`|$\displaystyle\int_{0}^{1} x \cdot \ln(x) \,\text{d}x$|$\displaystyle\frac{1}{4}$| `PASS` |
|`CAL7`|$\displaystyle\lim_{x\to\infty} \left(1+\frac{1}{x}\right)^x$|$e$ | `PASS`   |

### 5. Differential Equations

| Test Case No | Math Expression | Expected Output |              STATUS               |
| ------------ | --------------- | --------------- | --------------------------------- |
|`DE1`|$\displaystyle y' - 2y = 0$| $C_{1} \cdot e^{2x}$ | `PASS`                      |
|`DE2`|$\displaystyle y''+ 4y = 0$| $C_{1} \cos(2x) + C_{2} \sin(2x)$| `PASS`          |
|`DE3`|$\displaystyle y'' + y' + y = 0$| $\displaystyle C_1 \cdot e^{-\frac{x}{2}} \cdot \cos\left(\frac{\sqrt{3}}{2}x\right) + C_2 \cdot e^{-\frac{x}{2}} \cdot \sin\left(\frac{\sqrt{3}}{2}x\right)$| `PASS`             |

### 6. Series and Sequences

| Test Case No | Math Expression | Expected Output |             STATUS              |
| ------------ | --------------- | --------------- | ------------------------------- |
|`SER1`|$\displaystyle\sum_{n=1}^{\infty} \frac{1}{n^2}$| $\displaystyle \frac{\pi^2}{6}$ | `PASS` |
|`SER2`|$\displaystyle\sum_{n=0}^{\infty} \frac{1}{n!}$ | $e$               | `PASS` |

### 7. Number Theories

To Be Added

### 8. Linear Algebra

To Be Added