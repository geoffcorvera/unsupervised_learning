import numpy as np
from kmeans import KMeans

class FCM(object):


    def __init__(self,c,m,X):
        assert m > 1
        # initialize memberships, & scale so rows sum to 1
        memberships = np.random.rand(X.shape[0], c)
        sums = np.sum(memberships, axis=1)
        memberships = memberships/sums[:,np.newaxis]
        
        centroids = self.initCentroids(c,X)

        self.c = c
        self.m = m
        self.memberships = memberships
        self.centroids = centroids

    
    # Initialize centroids with a KMeans run
    def initCentroids(self,c,X):
        km = KMeans(c,X)
        km.train(X)
        return np.copy(km.centroids)
    

    # TODO: refactor janky loops into matrix ops with numpy
    # multiply 1D array elems to corresponding row in 2D array:
    # scaled_X = self.memberships[:,c:c+1] * fuzzy_scores[:,np.newaxis]
    def nextCentroid(self,X):
        nfeatures = X.shape[1]
        mem_raised = np.power(self.memberships, self.m)
        collector = np.zeros((1,nfeatures))

        for c in range(self.c):
            cluster_weights = mem_raised[:,c:c+1]
            for w,x in zip(cluster_weights, X):
                collector += w * x
            
            collector = collector / np.sum(cluster_weights)
            self.centroids[c] = collector
            collector = np.zeros((1,nfeatures))


    def nextMemberships(self,X):
        coef = float(2/(self.m-1))
        for i,xi in enumerate(X):
            distances = np.array([np.linalg.norm(xi-ck) for ck in self.centroids])
            for j,cj in enumerate(self.centroids):
                t = np.linalg.norm(xi-cj)**coef
                sumterm = np.sum((1/(distances))**coef)
                self.memberships[i][j] = 1/(t*sumterm)


    def fit(self,X,maxiter=10):
        curr = np.copy(self.centroids)
        prev = np.zeros(curr.shape)
        i = 0
        while not np.allclose(prev, curr) and i < maxiter:
            print(f'iteration: {i}')
            self.nextCentroid(X)
            self.nextMemberships(X)
            prev = np.copy(curr)
            curr = np.copy(self.centroids)
            i += 1

    # For each datum, returns cluster num with highest membership score
    def classify(self):
        return np.apply_along_axis(np.argmax, 1, self.memberships)
        