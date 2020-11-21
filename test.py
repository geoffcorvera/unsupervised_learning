import numpy as np
from kmeans import KMeans  

data = np.genfromtxt('data/545_cluster_dataset.txt')
labels = data[:,:1].astype(int)
training = data[:,1:]

km = KMeans(3,training)
km.train(training)
predictions = km.classify(training)
