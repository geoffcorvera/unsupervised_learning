import numpy as np
import matplotlib.pyplot as plt
from kmeans import KMeans 
from fcm import FCM

colors = np.array(['#49111c','#ee2e31','#1d7874','#7f7f7f'])
data = np.genfromtxt('data/545_cluster_dataset.txt')

def SSE(clusters, centroids):
    """
    For each point, the error is the distance to the nearest cluster
    SSE:
        for each cluster
            sum:
            for each datum in cluster
                sum: norm(k-centroid, xi)**2
    """
    pass

def runTrials(ntrials):
    pass

def plotGroundTruth(X):
    c1 = X[:500]
    c2 = X[500:1000]
    c3 = X[1000:]
    clusters = [c1,c2,c3]
    for i,cluster in enumerate(clusters):
        plt.scatter(cluster[:,:1], cluster[:,1:], c=colors[i])
    plt.show()

def generateTestData():
    cluster1 = np.random.normal(2, 0.3, size=(5,2))
    cluster2 = np.random.normal(3, 0.3, size=(5,2))
    cluster3 = np.random.normal((-1,3), 0.2, size=(5,2))
    return np.concatenate([cluster1, cluster2, cluster3], axis=0)

def km_test(K):
    km = KMeans(K,data)
    km.train(data)
    results = km.classify(data)

    # Split test data by cluster assignments
    clusters = [data[np.where(results==k)] for k in range(K)]

    # Plot results
    for cluster, color in zip(clusters, colors[:K]):
        plt.scatter(cluster[:,:1], cluster[:,1:], c=color)
    plt.show()

def fcm_test(K):
    fcm = FCM(K,1.2,data)
    fcm.fit(data, 40)
    results = fcm.classify()

    # Split test data by results
    splits = [data[np.where(results==k)] for k in range(K)]
    
    # Plot results
    for cluster,color in zip(splits,colors[:K]):
        plt.scatter(cluster[:,:1],cluster[:,1:], c=color)
    plt.show()

# Run test
# fcm_test(K=3)
plotGroundTruth(data)