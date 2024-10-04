import numpy as np
import matplotlib.pyplot as plt

# 1. Generate Synthetic Dataset
np.random.seed(0)
n_samples = 100
x = np.random.rand(n_samples) * 10
true_slope = 2
true_intercept = 3
noise = np.random.randn(n_samples) * 2
y = true_slope * x + true_intercept + noise

# 2. Compute Slope and Intercept Using Matrix Operations
X = np.vstack((np.ones(n_samples), x)).T
X_transpose = X.T
XTX = X_transpose @ X
XTX_inv = np.linalg.inv(XTX)
XTy = X_transpose @ y
theta = XTX_inv @ XTy
intercept, slope = theta
print(f"Computed intercept: {intercept:.4f}")
print(f"Computed slope: {slope:.4f}")

# 3. Perform Least Squares Solution Using NumPy
theta_ls, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
intercept_ls, slope_ls = theta_ls
print(f"Least Squares intercept: {intercept_ls:.4f}")
print(f"Least Squares slope: {slope_ls:.4f}")

# 4. Plot the Dataset and the Regression Line
x_reg = np.linspace(0, 10, 100)
y_reg = slope * x_reg + intercept

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x_reg, y_reg, color='red', label=f'Best Fit Line: y = {slope:.2f}x + {intercept:.2f}')
plt.xlabel('Feature (x)')
plt.ylabel('Target (y)')
plt.title('Simple Linear Regression using NumPy')
plt.legend()
plt.grid(True)
plt.show()