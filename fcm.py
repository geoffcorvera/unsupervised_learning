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
    
    def update_centroids(self, data):
        fuzzy_scores = np.power(self.member_scores, self.m)
        for c in range(self.c):
            scaled_X = self.member_scores[:,c:c+1] * fuzzy_scores[:,np.newaxis]
            self.centroids[c] = np.sum(scaled_X, axis=0)
    
# Randomly initialize membership scores for each data point
# Repeat until convergence or stopping condition
    # Compute centroid for each cluster (M-step)
    # For each data point, recompute membership scores for being in the clusters (E-step)

test_data = np.random.rand(4,2)
fcm = FCM(3, 1.2, test_data)
fcm.update_centroids(test_data)