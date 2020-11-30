import sys, random
import numpy as np

class KMeans(object):

    def __init__(self,k,X):
        nf = X.shape[1]
        # Get random sample from input X
        sample = [np.random.choice(X.ravel(),nf,replace=False) for _ in range(k)]
        self.centroids = np.array(sample)
        self.k = k
        self.nft = X.shape[1]

    # Returns total sum-of-squares error over all clusters
    def SSE(self,X):
        results = self.classify(X)
        assert len(results) == len(X)
        err = 0
        for k in range(self.k):
            cluster = X[np.where(results==k)]
            err += np.sum(np.linalg.norm(cluster-self.centroids[k], axis=1))
        return err

    # E step (assign most plausible cluster to xs)
    def assign(self, X):
        self.clusters = [list() for _ in range(self.k)]
        for x in X:
            deltas = [x - m for m in self.centroids]
            norms = [np.linalg.norm(d) for d in deltas]
            nearest = np.argmin(norms)
            self.clusters[nearest].append(x)
    
    # Estimate maximize likelihood of the paramters (recalculate k centroids)
    def updateParams(self):
        for cluster, k in zip(self.clusters, range(self.k)):
            cluster = np.array(cluster)
            if cluster.size > 0:
                # Calc mean of each feature
                centroid = np.mean(cluster,axis=0)
                self.centroids[k] = centroid
    
    def train(self, X, niter=100):
        prev = None
        curr = np.copy(self.centroids)
        i = 0
        err_per_epoch = list()

        # while not np.array_equal(prev, curr) and i < niter:
        while not np.array_equal(prev, curr) and i < niter:
            self.assign(X)
            self.updateParams()
            prev = np.copy(curr)
            curr = np.copy(self.centroids)
            err_per_epoch.append(self.SSE(X))
            i += 1
            
        print(f"k-means finished in {i+1} iterations")
        return err_per_epoch
            
    def classify(self,X):
        # Calculate distances from k cluster centroids
        d1 = np.linalg.norm(X-self.centroids[0], axis=1)
        d2 = np.linalg.norm(X-self.centroids[1], axis=1)
        d3 = np.linalg.norm(X-self.centroids[2], axis=1)
        # d4 = np.linalg.norm(X-self.centroids[3], axis=1)
        centroid_distances = np.c_[d1,d2,d3]
        return np.apply_along_axis(np.argmin, 1, centroid_distances)
