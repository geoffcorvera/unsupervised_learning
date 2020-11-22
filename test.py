import numpy as np
import matplotlib.pyplot as plt
from kmeans import KMeans 

data = np.genfromtxt('data/545_cluster_dataset.txt')
K = 3
km = KMeans(K,data)
km.train(data)
results = km.classify(data)

# Split clusters
clusters = [data[np.where(results==k)] for k in range(K)]

# Plot
colors = np.array(['r','g','b','#7f7f7f'])
for cluster, color in zip(clusters, colors[:K]):
    plt.scatter(cluster[:,:1], cluster[:,1:], c=color)
plt.show()
