import sys, random
import numpy as np

try:
    K = int(sys.argv[1])
except (IndexError):
    K = 2

class KMeans(object):
    def __init__(self,k):
        self.k = k
        self.ksets = [set() for _ in range(K)]
        self.centroids = list()

    def train(self):
        self.centroids = [random.random() for _ in range(self.k)]

    # argmin(range(k):for s in ksets)
    def classify(self,x):
        x = np.array(x)
        wcss = [np.linalg.norm(x-u) for u in self.centroids]
        return np.argmin(np.array(wcss))

nft = 4
km = KMeans(K)
km.train()
datum = [random.random() for _ in range(nft)]
print(km.classify(datum))
