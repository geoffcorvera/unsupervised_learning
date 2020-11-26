import numpy as np

class FCM(object):

    def __init__(self, c, dataset):
        nfeat = dataset.shape[1]
        centroids = np.random.rand((c, nfeat))
        
        self.c = c
        self.centroids = centroids
    
# Randomly initialize membership scores for each data point
# Repeat until convergence or stopping condition
    # Compute centroid for each cluster (M-step)
    # For each data point, recompute membership scores for being in the clusters (E-step)