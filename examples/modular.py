import numpy as np
import tensorflow as tf

import bayesflow as bf
import bayesflow.util as util

from bayesflow.models import FlatDistribution
from bayesflow.models.elementary import GaussianMatrix, BernoulliMatrix, BetaMatrix, DirichletMatrix
from bayesflow.models.q_distributions import DeltaQDistribution, GaussianQDistribution, BernoulliQDistribution, SimplexQDistribution
from bayesflow.models.matrix_decompositions import *
from bayesflow.models.transforms import PointwiseTransformedMatrix, PointwiseTransformedQDistribution
from bayesflow.models.neural import VAEEncoder, VAEDecoderBernoulli, init_weights, init_zero_vector
from bayesflow.models.train import optimize_elbo, print_inference_summary

"""
Examples / test cases for a new API allowing construction of
models and inference routines from modular components.
"""


def gaussian_mean_model():
    mu = GaussianMatrix(mean=0, std=10, output_shape=(1,))
    X = GaussianMatrix(mean=mu, std=1, output_shape=(100,))

    sampled_X = X.sample(seed=2)
    X.observe(sampled_X)

    return X
    
def gaussian_lowrank_model():
    A = GaussianMatrix(mean=0.0, std=1.0, output_shape=(100, 3), name="A")
    B = GaussianMatrix(mean=0.0, std=1.0, output_shape=(100, 3), name="B")
    C = NoisyGaussianMatrixProduct(A=A, B=B, std=0.1, name="noise")

    sampled_C = C.sample(seed=0)
    C.observe(sampled_C)

    return C
    
def gaussian_randomwalk_model():
    A = GaussianMatrix(mean=0.0, std=1.0, output_shape=(100, 2), name="A")
    C = NoisyCumulativeSum(A=A, std=0.1, name="noise")

    sampled_C = C.sample(seed=0)
    C.observe(sampled_C)

    return C

def clustering_gmm_model(n_clusters = 4,
                         cluster_center_std = 5.0,
                         cluster_spread_std = 2.0,
                         n_points = 500,
                         dim = 2):

    centers = GaussianMatrix(mean=0.0, std=cluster_center_std, output_shape=(n_clusters, dim), name="centers")

    weights = DirichletMatrix(alpha=1.0,
                              output_shape=(n_clusters,),
                              name="weights")


    X = GMMClustering(weights=weights, centers=centers,
                      std=cluster_spread_std, output_shape=(n_points, dim), name="noise")

    sampled_X = X.sample(seed=0)
    X.observe(sampled_X)

    return X

def latent_feature_model():
    K = 3
    D = 10
    N = 100

    a, b = np.float32(1.0), np.float32(1.0)

    pi = BetaMatrix(alpha=a, beta=b, output_shape=(K,), name="pi")
    B = BernoulliMatrix(p=pi, output_shape=(N, K), name="B")
    G = GaussianMatrix(mean=0.0, std=1.0, output_shape=(K, D), name="G")
    D = NoisyLatentFeatures(B=B, G=G, std=0.1, name="noise")

    sampled_D = D.sample(seed=0)
    D.observe(sampled_D)

    return D


def sparsity():
    G1 = GaussianMatrix(mean=0, std=1.0, output_shape=(100,10), name="G1")
    expG1 = PointwiseTransformedMatrix(G1, bf.transforms.exp, name="expG1")
    X = MultiplicativeGaussianNoise(expG1, 1.0, name="multGaussian")

    sampled_X = X.sample()
    X.observe(sampled_X)

    return X

def autoencoder():
    d_z = 2
    d_hidden=256
    d_x = 28*28
    N=100

    from util import get_mnist
    Xdata, ydata = get_mnist()
    Xbatch = Xdata[0:N]
    
    def init_decoder_params(d_z, d_hidden, d_x):
        # TODO come up with a simpler/more elegant syntax for point weights.
        # maybe just let the decoder initialize and manage its own weights / q distributions? 
        w_decode_h = DeltaQDistribution(init_weights(d_z, d_hidden))
        w_decode_h2 = DeltaQDistribution(init_weights(d_hidden, d_x))
        b_decode_1 = DeltaQDistribution(init_zero_vector(d_hidden))
        b_decode_2 = DeltaQDistribution(init_zero_vector(d_x))
        
        w1 = FlatDistribution(value=np.zeros((d_z, d_hidden), dtype=np.float32), fixed=False, name="w1")
        w1.attach_q(w_decode_h)
        w2 = FlatDistribution(value=np.zeros(( d_hidden, d_x), dtype=np.float32), fixed=False, name="w2")
        w2.attach_q(w_decode_h2)
        b1 = FlatDistribution(value=np.zeros((d_hidden,), dtype=np.float32), fixed=False, name="b1")
        b1.attach_q(b_decode_1)
        b2 = FlatDistribution(value=np.zeros((d_x), dtype=np.float32), fixed=False, name="b2")
        b2.attach_q(b_decode_2)
        
        return w1, w2, b1, b2

    w1, w2, b1, b2 = init_decoder_params(d_z, d_hidden, d_x)
    z = GaussianMatrix(mean=0, std=1.0, output_shape=(N,d_z), name="z")
    X = VAEDecoderBernoulli(z, w1, w2, b1, b2, name="X")
    
    X.observe(Xbatch)
    tfX = tf.constant(Xbatch, dtype=tf.float32)

    q_z = VAEEncoder(tfX, d_hidden, d_z)
    z.attach_q(q_z)
    
    return X

def main():


    print "gaussian mean estimation"
    obs_node = gaussian_mean_model()
    elbo_terms, posterior = optimize_elbo(obs_node)
    print_inference_summary(elbo_terms, posterior)


    print "gaussian matrix factorization"
    obs_node = gaussian_lowrank_model()
    elbo_terms, posterior = optimize_elbo(obs_node)
    print_inference_summary(elbo_terms, posterior)

    print "gaussian random walk"
    obs_node = gaussian_randomwalk_model()
    elbo_terms, posterior = optimize_elbo(obs_node)
    print_inference_summary(elbo_terms, posterior)

    
    print "gaussian mixture model"
    obs_node = clustering_gmm_model()
    elbo_terms, posterior = optimize_elbo(obs_node)
    print_inference_summary(elbo_terms, posterior)
    
    print "latent features"
    obs_node = latent_feature_model()
    elbo_terms, posterior = optimize_elbo(obs_node, steps=300)
    print_inference_summary(elbo_terms, posterior)


    print "bayesian sparsity"
    obs_node = sparsity()
    elbo_terms, posterior = optimize_elbo(obs_node)
    print_inference_summary(elbo_terms, posterior)

    print "variational autoencoder"
    obs_node = autoencoder()
    elbo_terms, posterior = optimize_elbo(obs_node, adam_rate=0.001)
    print_inference_summary(elbo_terms, posterior)
    
    
if __name__ == "__main__":
    main()
