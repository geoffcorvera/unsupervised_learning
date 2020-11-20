import sys, random
import numpy as np

try:
    K = int(sys.argv[1])
except (IndexError):
    K = 2

class KMeans(object):
    def __init__(self,k,X):
        # randomly initialize k centroids
        self.centroids = np.random.rand(k, X.shape[1])*np.max(X)
        self.k = k
        self.nft = X.shape[1]

    # E step (assign most plausible cluster to xs)
    def assign(self, X):
        self.clusters = [list() for _ in range(self.k)]
        for x in X:
            deltas = [x - m for m in self.centroids]
            norms = [np.linalg.norm(d) for d in deltas]
            nearest = np.argmin([np.linalg.norm(n) for n in norms])
            self.clusters[nearest].append(x)
    
    # Estimate maximize likelihood of the paramters (recalculate k centroids)
    def updateParams(self):
        for cluster, k in zip(self.clusters, range(self.k)):
            cluster = np.array(cluster)
            if cluster.size > 0:
                # Calc mean of each feature
                centroid = np.mean(cluster,axis=0)
                self.centroids[k] = centroid
    
    def train(self, X, niter=11):
        prev = None
        curr = np.copy(self.centroids)
        i = 0
        while not np.array_equal(prev, curr) and i < niter:
            print(f"iteration: {i}")
            self.assign(X)
            self.updateParams()
            prev = curr
            curr = self.centroids
            i += 1
            
    # XXX: use apply_along_axis(argmin(centroids),1,X)
    def classify(self,x):
        x = np.array(x)
        wcss = [np.linalg.norm(x-u) for u in self.centroids]
        return np.argmin(np.array(wcss))