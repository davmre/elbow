import numpy as np
import tensorflow as tf

import bayesflow as bf

from bayesflow.mean_field import MeanFieldInference

"""
Clustering using a Gaussian mixture model with fixed
variances. Demonstrates how to handle discrete variables (in this
case, cluster assignments) by marginalizing them out.
"""

def draw_prior_sample(n_points, n_clusters, dim, cluster_center_std, cluster_spread_std, seed=None):

    """
    Create synthetic data by sampling from the model
    """
    
    if seed is not None:
        np.random.seed(seed)

    # cluster means from an isotropic gaussian prior
    cluster_centers = np.random.randn(n_clusters, dim) * cluster_center_std

    points = []
    for i in range(n_points):
        # for each point, choose a cluster uniformly at random
        z = np.random.choice(np.arange(n_clusters))
        center = cluster_centers[z]

        # point is near cluster center, with isotropic gaussian noise
        pt = center + np.random.randn(dim) * cluster_spread_std 
        points.append(pt)
        
    points = np.array(points)
    return cluster_centers, points


def gmm_density_marginalized(centers, weights, points, cluster_center_std, cluster_spread_std):
    """
    the full GMM model p(C, Z, X) = p(C)p(Z)p(X | Z, C)
      cluster centers C with Gaussian prior
      cluster assignments Z with prior of form multinomial(weights) 
      observed points X from Gaussian conditionals

    we can write as p(C, X) = p(C) \sum_Z p(X, Z | C)
                            = p(C) \sum_Z p(Z) p(X | Z, C)
                            = p(C) \sum_Z \prod_i p(z_i) p(x_i | z_i, C)
                            = p(C) \prod_i \sum_{z_i} p(z_i) p(x_i | z_i, C)
    or in practice log p(C, X) = log p(C) + \sum_i log \sum_{z_i} weights[z_i] N(x_i; C_{z_i}, cluster_spread_std)

    where the point is that by marginalizing over the assignments Z, all remaining variables 
    are continuous and easy to optimize over. 

    If we didn't do this, we'd have to explicitly represent a variational posterior q(Z), presumably
    factored into a multinomial distribution for each point z_i. But this posterior is hard to
    update because we can't use the reparameterization trick.
    """
    # log p(C)
    prior_lp = tf.reduce_sum(bf.dists.gaussian_log_density(centers, 0.0, cluster_center_std))

    total_ps = None
    # loop over clusters
    for i, center in enumerate(tf.unpack(centers)):
        # compute vector of likelihoods that each point could be generated from *this* cluster
        cluster_lls = tf.reduce_sum(bf.dists.gaussian_log_density(points, center, cluster_spread_std), 1)

        # sum these likelihoods, weighted by cluster probabilities
        cluster_ps = weights[i] * tf.exp(cluster_lls)
        total_ps = total_ps + cluster_ps if total_ps is not None else cluster_ps

    # finally sum the log probabilities of all points to get a likelihood for the dataset
    obs_lp = tf.reduce_sum(tf.log(total_ps))
    
    return prior_lp + obs_lp


def main():

    n_clusters = 4
    cluster_center_std = 5.0
    cluster_spread_std = 2.0
    n_points = 500
    dim = 2

    true_centers, points = draw_prior_sample(n_points, n_clusters, dim,
                                             cluster_center_std, cluster_spread_std)

    mf = MeanFieldInference(gmm_density_marginalized,
                            points=points,
                            cluster_center_std=cluster_center_std,
                            cluster_spread_std=cluster_spread_std)
    
    init_centers = np.float32(np.random.randn(n_clusters, dim))
    mf.add_latent("centers", init_centers,
                  np.float32(np.ones((n_clusters,dim)) * 1e-3), None, shape=(n_clusters, dim))

    # learn a point estimate of the cluster weights. Note that we
    # parameterize the weights by an unconstrained vector that is then
    # transformed to lie in the simplex.
    init_weights = np.float32(np.zeros((n_clusters,)))
    mf.add_latent("weights", init_weights,
                  point_estimate=True,
                  transform=bf.transforms.simplex,
                  shape=(n_clusters, dim))

    elbo = mf.build_stochastic_elbo(n_eps=1)

    sess = tf.Session()
    mf.train(adam_rate=0.01, print_interval=50,
             steps=3000,
             display_dict={"density": mf.expected_density,},
             sess=sess)
    
    inferred_centers, inferred_weights = sess.run((mf.latents["centers"]["q_mean"], mf.latents["weights"]["q_mean"]))
    inferred_weights_transformed, _ = bf.transforms.simplex(inferred_weights)

    print "true cluster centers"
    print true_centers

    print "inferred cluster centers"
    print inferred_centers

    print "inferred cluster weights:", inferred_weights_transformed

if __name__ == "__main__":
    main()
    

