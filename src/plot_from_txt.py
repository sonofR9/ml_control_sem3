import matplotlib.pyplot as plt
import numpy as np

# Read data from files
data_x = np.loadtxt("trajectory_x.txt")
data_y = np.loadtxt("trajectory_y.txt")

# Create time axis
time = np.arange(len(data_x))

# Plot the trajectory
plt.plot(data_x, data_y, label="trajectory")

circle = plt.Circle((2.5, 2.5), np.sqrt(2.5), color='r')
plt.gca().add_patch(circle)
circle2 = plt.Circle((7.5, 7.5), np.sqrt(2.5), color='r')
plt.gca().add_patch(circle2)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Trajectory")
plt.legend()
plt.show()
