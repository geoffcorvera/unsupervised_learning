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
        # XXX breaks if self.m == 1
        coef = float(2/(self.m-1))
        for i,xi in enumerate(X):
            distances = np.array([np.linalg.norm(xi-ck) for ck in self.centroids])
            for j,cj in enumerate(self.centroids):
                t = np.linalg.norm(xi-cj)**coef
                sumterm = np.sum((1/(distances))**coef)
                self.memberships[i][j] = 1/(t*sumterm)


    def fit(self,X,maxiter=10):
        prev = None
        curr = np.copy(self.centroids)
        i = 0
        while not np.array_equal(prev, curr) and i < maxiter:
            print(f'iteration: {i}')
            self.nextCentroid(X)
            self.nextMemberships(X)
            prev = np.copy(curr)
            curr = np.copy(self.centroids)
            i += 1


    def classify(self, X):
        pass