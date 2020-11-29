import numpy as np

class FCM(object):

    def __init__(self,c,m,X):
        # initialize memberships, & scale so rows sum to 1
        memberships = np.random.rand(X.shape[0], c)
        sums = np.sum(memberships, axis=1)
        memberships = memberships/sums[:,np.newaxis]
        
        # XXX: Should initial centroids be scaled and shifted?
        centroids = np.random.rand(c, X.shape[1])

        self.c = c
        self.m = m
        self.memberships = memberships
        self.centroids = centroids
    
    def nextCentroid(self):
        pass

    def nextMemberships(self,X):
        # XXX breaks if self.m == 1
        coef = float(2/(self.m-1))
        for i,xi in enumerate(X):
            distances = np.array([np.linalg.norm(xi-ck) for ck in self.centroids])
            for j,cj in enumerate(self.centroids):
                t = np.linalg.norm(xi-cj)**coef
                sumterm = np.sum((1/(distances))**coef)
                self.memberships[i][j] = 1/(t*sumterm)


    def fit(self, X):
        pass

    def classify(self, X):
        pass