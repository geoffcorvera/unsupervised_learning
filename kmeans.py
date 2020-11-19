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

    def assign(self, X):
        self.clusters = [list() for _ in range(self.k)]
        for x in X:
            deltas = [x - m for m in self.centroids]
            norms = [np.linalg.norm(d) for d in deltas]
            nearest = np.argmin([np.linalg.norm(n) for n in norms])
            self.clusters[nearest].append(tuple(x))

    def train(self):
        self.centroids = [random.random() for _ in range(self.k)]

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
    kmc.train()
    print(kmc.clusters)

test(k=3, data=np.random.randint(1,5,(3,2)))