import numpy as np
from kmeans import KMeans

class FCM(object):

    # Initialize model parameters
    def __init__(self,c,m,X):
        assert m > 1
        # initialize memberships, & scale so rows sum to 1
        memberships = np.random.rand(X.shape[0], c)
        sums = np.sum(memberships, axis=1)
        memberships = memberships/sums[:,np.newaxis]
        
        # Initialize centroids using k-means
        km = KMeans(c,X)
        km.assign(X)
        km.updateParams()
        centroids = np.copy(km.centroids)

        self.c = c
        self.m = m
        self.memberships = memberships
        self.centroids = centroids
    
    
    # Recalculate centroids
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


    # Recalculate membership scores for each datum
    def nextMemberships(self,X):
        coef = float(2/(self.m-1))
        for i,xi in enumerate(X):
            distances = np.array([np.linalg.norm(xi-ck) for ck in self.centroids])
            for j,cj in enumerate(self.centroids):
                t = np.linalg.norm(xi-cj)**coef
                sumterm = np.sum((1/(distances))**coef)
                self.memberships[i][j] = 1/(t*sumterm)


    # Calculates the sum squared err for each cluster and returns total SSE
    def SSE(self,X):
        results = self.classify()
        assert len(results) == len(X)
        
        err = 0
        for c in range(self.c):
            cluster = X[np.where(results==c)]
            err += np.sum(np.linalg.norm(cluster-self.centroids[c], axis=1))

        return err
            
    
    def train(self,X,maxiter=50):
        curr = np.copy(self.centroids)
        prev = np.zeros(curr.shape)
        i = 0
        err_per_epoch = list()
        
        while not np.allclose(prev, curr) and i < maxiter:
            self.nextCentroid(X)
            self.nextMemberships(X)

            prev = np.copy(curr)
            curr = np.copy(self.centroids)
            err_per_epoch.append(self.SSE(X))
            i += 1
        
        print(f'  Finished in {i} iterations')
        return err_per_epoch


    # For each datum, returns cluster num with highest membership score
    def classify(self):
        return np.apply_along_axis(np.argmax, 1, self.memberships)
        