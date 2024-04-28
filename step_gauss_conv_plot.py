import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Initialize y with default values
y = np.array([0.2, 0.7, 0.3, 0.9, 0.5])

# Initialize sigma with a default value
sigma = 1.0

# Create the figure and axes
fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(bottom=0.3)  # Adjust bottom margin for sliders
ax.set_xlim(-0.5, 4.5)
ax.set_ylim(0, 1)

# Plot the points
points, = ax.plot(range(5), y, 'bo', label='Points')

# Plot the step function
x_step = np.linspace(-0.5, 4.5, 1000)
y_step = np.zeros_like(x_step)
for i in range(len(x_step)):
    if 0 <= int(x_step[i] + 0.5) < 5:
        y_step[i] = y[int(x_step[i] + 0.5)]
step_func, = ax.plot(x_step, y_step, 'orange', label='Step Function')

# Plot the Gaussian
x_gaussian = np.linspace(-2, 7, 1000)
def gaussian(x, mu, sigma):
    return (1 / (np.sqrt(2 * np.pi * sigma**2))) * np.exp(-((x - mu)**2) / (2 * sigma**2))

y_gaussian = gaussian(x_gaussian, 2.5, sigma)
# gaussian_plot, = ax.plot(x_gaussian, y_gaussian, 'g--', label='Gaussian')

# Plot the convolution result
y_convolved = np.convolve(y_step, y_gaussian, mode='same')
y_convolved_normalized = y_convolved / np.sum(y_gaussian)
convolved_plot, = ax.plot(x_step, y_convolved_normalized, 'purple', label='Convolution Result')

# Add labels and legend
ax.legend()

# Create sliders for y values
slider_y = []
slider_positions = np.linspace(0.2, 0.05, 6)  # Adjust slider positions
for i in range(5):
    ax_slider = plt.axes([0.1, slider_positions[i], 0.65, 0.03])  # Adjust slider size
    slider = Slider(ax_slider, f'y{i+1}', 0, 1, valinit=y[i])
    slider_y.append(slider)

# Create a slider for sigma
ax_sigma = plt.axes([0.1, slider_positions[-1], 0.65, 0.03])  # Position sigma slider at the bottom
slider_sigma = Slider(ax_sigma, 'Sigma', 0.1, 2, valinit=sigma)

# Update plot based on slider values
def update(val):
    for i in range(5):
        y[i] = slider_y[i].val
    sigma = slider_sigma.val
    
    points.set_ydata(y)
    
    for i in range(len(x_step)):
        if 0 <= int(x_step[i] + 0.5) < 5:
            y_step[i] = y[int(x_step[i] + 0.5)]
    step_func.set_ydata(y_step)
    
    y_gaussian = gaussian(x_gaussian, 2.5, sigma)
    # gaussian_plot.set_ydata(y_gaussian)
    
    y_convolved = np.convolve(y_step, y_gaussian, mode='same')
    y_convolved_normalized = y_convolved / np.sum(y_gaussian)
    convolved_plot.set_ydata(y_convolved_normalized)
    
    fig.canvas.draw_idle()

# Connect sliders to update function
for slider in slider_y:
    slider.on_changed(update)
slider_sigma.on_changed(update)

# Display the plot
plt.show()
