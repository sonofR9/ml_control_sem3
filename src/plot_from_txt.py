import matplotlib.pyplot as plt
import numpy as np

# Read data from files
data_x = np.loadtxt("trajectory_x.txt")
data_y = np.loadtxt("trajectory_y.txt")

# Create time axis
time = np.arange(len(data_x))

# Plot the trajectory
plt.plot(time, data_x, label="X")
plt.plot(time, data_y, label="Y")
plt.xlabel("Time")
plt.ylabel("Position")
plt.title("Trajectory")
plt.legend()
plt.show()
