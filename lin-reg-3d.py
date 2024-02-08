import numpy as np
import matplotlib.pyplot as plt

# Generate random data in 3D space
X = np.random.randint(0, 101, size=(100, 1))
Y = np.random.randint(0, 101, size=(100, 1))
Z = 2 * X + 3 * Y + 100 + 10 * np.random.randn(100, 1)

# Initialize weights and bias
A = 0
B = 0
C = 1

# Learning rate and number of iterations
learning_rate = 0.0001
iterations = 100000

# Training loop
for i in range(iterations):
    # Forward pass
    Z_pred = A * X + B * Y + C

    # Compute loss (mean squared error)
    loss = np.mean((Z_pred - Z) ** 2)

    # Backpropagation
    dA = (1 / len(X)) * np.sum(X * (Z_pred - Z))
    dB = (1 / len(X)) * np.sum(Y * (Z_pred - Z))
    dC = (1 / len(X)) * np.sum(Z_pred - Z)

    # Update weights and bias
    A -= learning_rate * dA
    B -= learning_rate * dB
    C -= learning_rate * dC

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the data points
ax.scatter(X, Y, Z, label='Data Points', c='blue', marker='o')

# Create a meshgrid for the regression surface
X_surface, Y_surface = np.meshgrid(np.linspace(0, 100, 100), np.linspace(0, 100, 100))
Z_surface = A * X_surface + B * Y_surface + C

# Plot the regression surface
ax.plot_surface(X_surface, Y_surface, Z_surface, color='red', alpha=0.7, label='Regression Surface')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title('3D Linear Regression')
plt.show()

# Print the final weights and bias
print(f'Final Weight A: {A}, Final Weight B: {B}, Final Bias C: {C}')
