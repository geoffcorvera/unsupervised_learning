import numpy as np
import matplotlib.pyplot as plt
from kmeans import KMeans 
from fcm import FCM

def km_test():
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

def fcm_test():
    # Setup test data
    cluster1 = np.random.normal(2, 0.3, size=(5,2))
    cluster2 = np.random.normal(3, 0.3, size=(5,2))
    cluster3 = np.random.normal((-1,3), 0.2, size=(5,2))
    test_data = np.concatenate([cluster1, cluster2, cluster3], axis=0)

    fcm = FCM(3,2,test_data)
    fcm.nextCentroid(test_data)
    fcm.nextMemberships(test_data)

fcm_test()