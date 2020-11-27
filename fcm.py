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
    

# Repeat until convergence or stopping condition
    # Compute centroid for each cluster (M-step)
    # For each data point, recompute membership scores for being in the clusters (E-step)

test_data = np.random.rand(4,2)
fcm = FCM(3, 1.2, test_data)
fcm.c_update(test_data)
print(fcm.centroids)