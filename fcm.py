import numpy as np

class FCM(object):

    def __init__(self, c, m, dataset):
        ndatums = dataset.shape[0]
        nfeat = dataset.shape[1]
        centroids = np.random.rand(c, nfeat)
        membership_scores = np.random.rand(ndatums, c)
        
        self.c = c                              # no. of clusters
        self.m = m                              # fuzzifier parameter
        self.centroids = centroids              # randomly initialized centroids
        self.member_scores = membership_scores  # randomly initialized membership scores
    
    # Updates centroids
    def c_update(self, data):
        # TODO: refactor janky loops into matrix ops with numpy
            # multiply 1D array elems to corresponding row in 2D array:
            # scaled_X = self.member_scores[:,c:c+1] * fuzzy_scores[:,np.newaxis]
        nfeatures = data.shape[1]
        weights = np.power(self.member_scores, self.m)
        collector = np.zeros((1,nfeatures))

        for c in range(self.c):
            cluster_weights = weights[:,c:c+1]
            for w,x in zip(cluster_weights, data):
                collector += w * x
            
            collector = collector / np.sum(cluster_weights)
            self.centroids[c] = collector
            collector = np.zeros((1,nfeatures))
    
    
    def classify(self, data):
        # Calculate distances from k cluster centroids
        d1 = np.linalg.norm(X-self.centroids[0], axis=1)
        d2 = np.linalg.norm(X-self.centroids[1], axis=1)
        d3 = np.linalg.norm(X-self.centroids[2], axis=1)
        centroid_distances = np.c_[d1,d2,d3]
        return np.apply_along_axis(np.argmin, 1, centroid_distances)

# Repeat until convergence or stopping condition
    # Compute centroid for each cluster (M-step)
    # For each data point, recompute membership scores for being in the clusters (E-step)

# Setup test data
cluster1 = np.random.normal(2, 0.3, size=(25,2))
cluster2 = np.random.normal(3, 0.3, size=(25,2))
cluster3 = np.random.normal((-1,3), 0.2, size=(25,2))
test_data = np.concatenate([cluster1, cluster2, cluster3], axis=0)

fcm = FCM(3, 1.2, test_data)
fcm.c_update(test_data)

import matplotlib.pyplot as plt

# Show test data
colors = ['#477998', '#A3333D', '#C4D6B0']
plt.scatter(test_data[:,:1], test_data[:,1:], c=colors[0])
plt.membership_update(test_data)
# plt.show()