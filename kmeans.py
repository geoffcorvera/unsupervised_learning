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

    def assign(self, X):
        self.clusters = [list() for _ in range(self.k)]
        for x in X:
            deltas = [x - m for m in self.centroids]
            norms = [np.linalg.norm(d) for d in deltas]
            nearest = np.argmin([np.linalg.norm(n) for n in norms])
            self.clusters[nearest].append(x)
    
    def update(self):
        for cluster, k in zip(self.clusters, range(self.k)):
            cluster = np.array(cluster)
            if cluster.size > 0:
                # Calc mean of each feature
                centroid = np.mean(cluster,axis=0)
                self.centroids[k] = centroid
    
    def train(self, X, niter=11):
        prev = None
        curr = self.centroids[:]
        for _ in range(niter):
            self.assign(X)
            self.update()
            print(self.centroids)

    def classify(self,x):
        x = np.array(x)
        wcss = [np.linalg.norm(x-u) for u in self.centroids]
        return np.argmin(np.array(wcss))

# E step (assign Z clusters)
    # find most plausible assignments for Z (sub-optimzation prob)
# MLE estimates (maximize likelihood of the paramters)
    # (MLE)est.params
    # k means parameters: cluster centroids (mus)
# repeat E & MLE]

def test(k,data):
    kmc = KMeans(k, data)
    kmc.assign(data)
    print(kmc.centroids)
    kmc.update()
    print(kmc.centroids)

test(k=3, data=np.random.randint(1,5,(30,2)))