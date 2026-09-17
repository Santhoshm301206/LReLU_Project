import numpy as np
import matplotlib.pyplot as plt

# Input values
values = np.array([-5, -3, -1, 0, 1, 3, 5])

# LReLU parameter
alpha = 0.01

# LReLU calculation
lrelu_output = np.where(values >= 0, values, alpha * values)

# ReLU calculation for comparison
relu_output = np.maximum(0, values)

# Display results
print("Input Values:")
print(values)

print("\nLReLU Output:")
print(lrelu_output)

print("\nReLU Output:")
print(relu_output)

# Graph
plt.figure(figsize=(8, 5))

plt.plot(values, lrelu_output, marker='o', label='LReLU')
plt.plot(values, relu_output, marker='o', label='ReLU')

plt.xlabel("Input Value")
plt.ylabel("Output Value")
plt.title("LReLU vs ReLU")
plt.legend()
plt.grid(True)

plt.show()