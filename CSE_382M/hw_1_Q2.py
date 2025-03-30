# import required libraries
import numpy as np
import matplotlib.pyplot as plt

num_of_points = 100

# generate points on unit sphere of d=3
points = np.random.randn(num_of_points, 3)
norms = np.linalg.norm(points, axis=1)
points = points / norms[:, None]

# compute the total distance between all points
total_distance = 0
for i in range(num_of_points):
    for j in range(i+1, num_of_points):
        total_distance += np.linalg.norm(points[i] - points[j])


# generate points on unit sphere of d=100
points_100 = np.random.randn(num_of_points, 100)
norms_100 = np.linalg.norm(points_100, axis=1)
points_100 = points_100 / norms_100[:, None]

# compute the total distance between all points
total_distance_100 = 0
for i in range(num_of_points):
    for j in range(i+1, num_of_points):
        total_distance_100 += np.linalg.norm(points_100[i] - points_100[j])



# plot histogram of total distances comparing d=3 and d=100
plt.bar(['d=3', 'd=100'], [total_distance, total_distance_100])
plt.legend()
plt.show()
