import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

# Create the figure and a main axis
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.3)  # leave space for the slider

# Initial distance value for the circle radius
init_distance = 8

# Draw a circle centered at (0, 0) with the initial radius
circle = plt.Circle((0, 0), init_distance, fill=False, color='blue')
ax.add_artist(circle)

# Set limits and equal aspect ratio
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.set_aspect('equal', 'box')
plt.title("Interactive Distance Modifier")

# Create a slider axis and slider widget for adjusting distance
ax_distance = plt.axes([0.25, 0.1, 0.65, 0.03])
distance_slider = Slider(ax_distance, 'Distance', 1, 20, valinit=init_distance)

def update_distance(val):
    new_distance = distance_slider.val
    circle.set_radius(new_distance)
    fig.canvas.draw_idle()  # Update the figure

distance_slider.on_changed(update_distance)

plt.show()
