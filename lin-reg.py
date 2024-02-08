import numpy as np
import matplotlib.pyplot as plt

# Tilfeldige punkter hvor X∈[0-100]
X = np.random.randint(0, 101, size=(100, 1))
y = 2 * X + 50 + 10 * np.random.randn(100, 1)

# Startverdier for A og B
A = 0
B = 1

# Endringsvilke og antall iterasjoner
learning_rate = 0.0005
iterations = 100000

for i in range(iterations):
    Y_pred = X.dot(A) + B
    
    dA = (1/len(X)) * X.T.dot(Y_pred - y)
    dB = (1/len(X)) * np.sum(Y_pred - y)

    A -= learning_rate * dA
    B -= learning_rate * dB

# Plot the data points
plt.scatter(X, y, label='Data Points', color='blue')

# Plot the regression line
y_regression = X.dot(A) + B
plt.plot(X, y_regression, label='y = %.2f x + %.2f' %(A, B), color='red')

plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('2D Linear Regression')
plt.show()

# Print the final weight and bias
print(f'Final Weight (w): {A}, Final Bias (b): {B}')
