import sys, random
import numpy as np

try:
    K = int(sys.argv[1])
except (IndexError):
    K = 2

class KMeans(object):
    def __init__(self,k,X):
        self.k = k
        # TODO: randomly initialize centroids (shift by X.mean, scale by X.std)
        self.centroids = list()

    def assign(self, X):
        self.clusters = [set() for _ in range(self.k)]
        for x in X:
            l2distance = [x - m for m in self.centroids]
            nearest = np.argmin([np.linalg.norm(d) for d in l2distance])
            self.clusters[nearest].add(x)
            self.centroids[nearest] = np.mean(self.clusters[nearest])

    def train(self):
        self.centroids = [random.random() for _ in range(self.k)]

    # argmin(range(k):for s in ksets)
    def classify(self,x):
        x = np.array(x)
        wcss = [np.linalg.norm(x-u) for u in self.centroids]
        return np.argmin(np.array(wcss))

nft = 4
X = np.ones((5,3))
km = KMeans(K, X)
km.train()
datum = [random.random() for _ in range(nft)]
print(km.classify(datum))

# E step (assign Z clusters)
    # find most plausible assignments for Z (sub-optimzation prob)
# MLE estimates (maximize likelihood of the paramters)
    # (MLE)est.params
    # k means parameters: cluster centroids (mus)
# repeat E & MLE]
