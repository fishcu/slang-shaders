import numpy as np
import matplotlib.pyplot as plt

def smooth(x1, x2, s):
    # y1 = x1 + ((x1 + x2) / 2 - x1) * (1 - (1 - s) * np.exp(-np.abs(x1 - x2)))
    # y2 = x2 + ((x1 + x2) / 2 - x2) * (1 - (1 - s) * np.exp(-np.abs(x1 - x2)))
    w = 1 - np.exp(-s * np.exp(-np.abs(x1 - x2)))
    avg = (x1 + x2) / 2
    y1 = (1 - w) * x1 + w * avg
    y2 = (1 - w) * x2 + w * avg
    return y1, y2

# Create a range of input values
x_vals = np.linspace(0, 10, 100)
x1_vals = x_vals
x2_vals = 10 - x_vals

# Define different smoothing strengths to plot
smoothing_strengths = [1, 2, 5, 100]

# Create a plot
plt.figure(figsize=(10, 6))

# Plot the input values
plt.plot(x_vals, x1_vals, 'k--', label='Input x1')
plt.plot(x_vals, x2_vals, 'k:', label='Input x2')

# Plot the smoothed values for each smoothing strength
for s in smoothing_strengths:
    y1_vals, y2_vals = smooth(x1_vals, x2_vals, s)
    plt.plot(x_vals, y1_vals, label=f'Smoothed y1 (s={s})')
    plt.plot(x_vals, y2_vals, label=f'Smoothed y2 (s={s})')

# Add labels and legend
plt.xlabel('x')
plt.ylabel('Value')
plt.title('Smoothing Function Behavior')
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()
