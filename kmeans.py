# K-Means Clustering on 2D points
# Computer Vision - Task

import numpy as np
import matplotlib.pyplot as plt

# 1) Create some points (3 groups)
np.random.seed(0)
group1 = np.random.randn(50, 2) + [2, 2]
group2 = np.random.randn(50, 2) + [8, 8]
group3 = np.random.randn(50, 2) + [2, 8]
points = np.vstack((group1, group2, group3))

# 2) Choose number of clusters
k = 3

# 3) Pick k random points as the first centers
indexes = np.random.choice(len(points), k, replace=False)
centers = points[indexes]

# 4) Repeat the steps
for i in range(10):
    # a) Assign every point to the nearest center
    labels = []
    for p in points:
        distances = np.linalg.norm(p - centers, axis=1)
        labels.append(np.argmin(distances))
    labels = np.array(labels)

    # b) Move each center to the mean of its points
    for c in range(k):
        centers[c] = points[labels == c].mean(axis=0)

# 5) Show the result
colors = ["red", "green", "blue"]
for c in range(k):
    cluster_points = points[labels == c]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[c], label="Cluster " + str(c + 1))

plt.scatter(centers[:, 0], centers[:, 1], color="black", marker="x", s=150, label="Centers")
plt.title("K-Means Clustering")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
