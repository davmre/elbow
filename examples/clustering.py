import tensorflow as tf
import numpy as np

from elbow import Model, Gaussian, BernoulliMatrix, BetaMatrix, DirichletMatrix
from elbow.models.factorizations import GMMClustering

def clustering_gmm_model(n_clusters = 4,
                         cluster_center_std = 5.0,
                         cluster_spread_std = 2.0,
                         n_points = 500,
                         dim = 2):

    centers = Gaussian(mean=0.0, std=cluster_center_std, shape=(n_clusters, dim), name="centers")
    weights = DirichletMatrix(alpha=1.0,
                              shape=(n_clusters,),
                              name="weights")
    X = GMMClustering(weights=weights, centers=centers,
                      std=cluster_spread_std, shape=(n_points, dim), name="X")

    jm = Model(X)
    return jm

def main():
    jm = clustering_gmm_model()

    sampled = jm.sample()
    jm["X"].observe(sampled["X"])
    
    jm.train()
    posterior = jm.posterior()

    weights = np.exp(posterior["q_weights"]["mean"])
    weights /= np.sum(weights)
    print "sampled cluster weights", sampled["weights"]
    print "inferred weights", weights

    print "sampled cluster centers", sampled["centers"]
    print "inferred cluster centers", posterior["q_centers"]["mean"]
    
if __name__ == "__main__":
    main()

