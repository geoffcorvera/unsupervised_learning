import numpy as np
import sys
import matplotlib.pyplot as plt
from kmeans import KMeans 
from fcm import FCM

colors = np.array(['#49111c','#ee2e31','#1d7874','#7f7f7f','#050517','#231651','#ff8484'])
data = np.genfromtxt('data/545_cluster_dataset.txt')

def plotTargetAssignments(X):
    c1 = X[:500]
    c2 = X[500:1000]
    c3 = X[1000:]
    clusters = [c1,c2,c3]
    for i,cluster in enumerate(clusters):
        plt.scatter(cluster[:,:1], cluster[:,1:], c=colors[i])
    plt.show()


def plotKClusters(model,k,X):
    results = model.classify(X)
    clusters = [X[np.where(results==i)] for i in range(k)]
    for cluster,color in zip(clusters,colors[:k]):
        plt.scatter(cluster[:,:1], cluster[:,1:], c=color)
    # Show centroids
    plt.scatter(model.centroids[:,:1], model.centroids[:,1:], marker='X',c='y')

# Runs r # of trials and selects model with lowest SSE
def kmeans_trials(k=3,r=1):

    # Create and train r models for trials
    models = [KMeans(k,data) for _ in range(r)]
    training_err = [m.train(data) for m in models]
    
    # Sort modes by sum-of-squares error
    results = [(err[-1], model) for err, model in zip(training_err,models)]
    results = sorted(results, key=lambda x: x[0])   # Sort asscending by sum square error

    # Plot trial results
    for i,trial in enumerate(results):
        final_err = round(trial[0], 2)
        m = trial[1]

        plt.title(f'Trial {i+1} Cluster Assignments (SSE={final_err})')
        plotKClusters(m, k, data)
        plt.show()
    
    # Show best model from r trials
    best_sse = round(results[0][0],2)
    best_model = results[0][1]
    plt.title(f"Best model (SSE={best_sse})")
    plotKClusters(best_model,k,data)
    plt.show()


# Runs n trials, and returns model with lowest sum-of-squares
# error, or models from all trials.
# TODO: remove "bestOnly" functionality
# TODO: refactor to make more like kmeans_trials
def fcm_trials(k,m=1.2,ntrials=5,bestOnly=True):
    models = list()
    SSEs = list()
    for i in range(ntrials):
        print(f'\nTRIAL {i+1}:')
        model = FCM(k,m,data)
        err = model.train(data)
        SSEs.append(err.pop())  # record final sum of squares error
        models.append(model)

        plt.title(f'FCM Trial {i+1}:')
        plt.suptitle(f'sum of squares error = {err.pop()}')
        plotKClusters(model,k,data)

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

# Select & run experiments
try:
    algo = sys.argv[1]
    K = int(sys.argv[2])
    try:
        R = int(sys.argv[3])
    except(IndexError):
        R = 1
    
    if algo=="km":
        kmeans_trials(K, R)
    elif algo=="fcm":
        m = float(sys.argv[4])
        fcm_trials(K,m,R)
    else:
        print("Incorrect algo. Select either: km (k-means) or fcm (fuzzy c-means)")
except(IndexError):
    print('Usage:\n  python3 test.py <"km"=kmeans, "fcm"=fuzzy c-means> <k=number of clusters> <r=number of trials> <m=fuzzifier (FOR FCM ONLY)>')
