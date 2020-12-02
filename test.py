import numpy as np
import matplotlib.pyplot as plt
from kmeans import KMeans 
from fcm import FCM

colors = np.array(['#49111c','#ee2e31','#1d7874','#7f7f7f'])
data = np.genfromtxt('data/545_cluster_dataset.txt')

def plotTargetAssignments(X):
    c1 = X[:500]
    c2 = X[500:1000]
    c3 = X[1000:]
    clusters = [c1,c2,c3]
    for i,cluster in enumerate(clusters):
        plt.scatter(cluster[:,:1], cluster[:,1:], c=colors[i])
    plt.show()


def plotKClusters(results,k,X):
    clusters = [X[np.where(results==i)] for i in range(k)]
    for cluster,color in zip(clusters,colors[:k]):
        plt.scatter(cluster[:,:1], cluster[:,1:], c=color)
    plt.show()


# TODO: run r trials and select model with lowest SSE
def kmeans_trials(k=3,r=1):
    # Create and fit model
    km = KMeans(k,data)
    err_plot = km.train(data)

    # Plot training results
    plt.plot(err_plot)
    plt.title(f"Sum-Squared-Error per Epoch (Final: {err_plot[-1]})")
    plt.show()

    # Plot final results
    results = km.classify(data)
    plt.title("Final Cluster Assignments")
    plotKClusters(results,k,data)

# Runs n trials, and returns model with lowest sum-of-squares
# error, or models from all trials.
def fcm_trials(k,m=1.2,ntrials=5,bestOnly=True):
    models = list()
    SSEs = list()
    for i in range(ntrials):
        print(f'\nTRIAL {i+1}:')
        model = FCM(k,m,data)
        err = model.train(data)
        SSEs.append(err.pop())  # record final sum of squares error
        models.append(model)
        # TODO: plot model results with appropriate trial label
        res = model.classify()
        plt.title(f'FCM Trial {i+1}:')
        plt.suptitle(f'sum of squares error = {err.pop()}')
        plotKClusters(res,k,data)

    if bestOnly:
        lowest_err = np.argmin(np.array(SSEs))
        return models[lowest_err]
    else:
        return models

def run_fcm_trials(): 
    # Run 10 fuzzy c-means trials and plot results of model with lowest SSE
    best_model = fcm_trials(k=3,m=1.2,ntrials=10)
    results = best_model.classify()
    plotKClusters(results,3,data)


# run_fcm_trials()
kmeans_trials()