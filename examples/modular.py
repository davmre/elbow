import numpy as np
import tensorflow as tf

import bayesflow as bf

from bayesflow.joint_model import Model
from bayesflow.elementary import Gaussian, BernoulliMatrix, BetaMatrix, DirichletMatrix
from bayesflow.transforms import DeterministicTransform, Exp
from bayesflow.models.neural import NeuralGaussian, NeuralBernoulli
from bayesflow.models.factorizations import *
"""
Examples / test cases for a new API allowing construction of
models and inference routines from modular components.
"""

def gaussian_mean_model():

    mu = Gaussian(mean=0, std=10, shape=(1,), name="mu")
    X = Gaussian(mean=mu, std=1, shape=(100,), name="X")

    sampled_X = X.sample(seed=0)
    X.observe(sampled_X)

    jm = Model(X)
    return jm
    
def gaussian_lowrank_model():

    A = Gaussian(mean=0.0, std=1.0, shape=(100, 3), name="A")
    B = Gaussian(mean=0.0, std=1.0, shape=(100, 3), name="B")
    C = NoisyGaussianMatrixProduct(A=A, B=B, std=0.1, name="C")


    sampled_C = C.sample(seed=0)
    C.observe(sampled_C)

    jm = Model(C)
    return jm
    
def gaussian_randomwalk_model():

    A = Gaussian(mean=0.0, std=1.0, shape=(100, 2), name="A")
    C = NoisyCumulativeSum(A=A, std=0.1, name="C")

    sampled_C = C.sample(seed=0)
    C.observe(sampled_C)
    jm = Model(C)
    
    return jm

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

    sampled_X = X.sample(seed=0)
    X.observe(sampled_X)

    jm = Model(X)
    return jm

def latent_feature_model():
    K = 3
    D = 10
    N = 100

    a, b = np.float32(1.0), np.float32(1.0)

    pi = BetaMatrix(alpha=a, beta=b, shape=(K,), name="pi")
    B = BernoulliMatrix(p=pi, shape=(N, K), name="B")
    G = Gaussian(mean=0.0, std=1.0, shape=(K, D), name="G")
    D = NoisyLatentFeatures(B=B, G=G, std=0.1, name="D")
        
    sampled_D = D.sample(seed=0)
    D.observe(sampled_D)
    jm = Model(D)

    return jm


def sparsity():
    G1 = Gaussian(mean=0, std=1.0, shape=(100,10), name="G1")
    expG1 = DeterministicTransform(G1, Exp, name="expG1")
    X = MultiplicativeGaussianNoise(expG1, 1.0, name="X")

    sampled_X = X.sample(seed=0)
    X.observe(sampled_X)

    jm = Model(X)
    
    return jm

def autoencoder():
    d_z = 2
    d_hidden=256
    d_x = 28*28
    N=100

    from util import get_mnist
    Xdata, ydata = get_mnist()
    Xbatch = tf.constant(np.float32(Xdata[0:N]))

    z = Gaussian(mean=0, std=1.0, shape=(N,d_z), name="z")
    X = NeuralBernoulli(z, d_hidden=d_hidden, d_x=d_x, name="X")

    X.observe(Xbatch)
    q_z = NeuralGaussian(X=Xbatch, d_hidden=d_hidden, d_z=d_z, name="q_z")
    z.attach_q(q_z)

    jm = Model(X)
    
    return jm

def main():

    print "gaussian mean estimation"
    model = gaussian_mean_model()
    posterior = model.train(steps=500)
    print posterior

    print "gaussian matrix factorization"
    model = gaussian_lowrank_model()
    posterior = model.train(steps=500)
    print posterior

    print "gaussian random walk"
    model = gaussian_randomwalk_model()
    posterior = model.train(steps=1000)
    print posterior
    
    print "gaussian mixture model"
    model = clustering_gmm_model()
    posterior = model.train(steps=1000)
    print posterior

    print "latent features"
    model = latent_feature_model()
    posterior = model.train(steps=1000)
    print posterior

    print "bayesian sparsity"
    model = sparsity()
    posterior = model.train(steps=1000)
    print posterior
    
    print "variational autoencoder"
    model = autoencoder()
    posterior = model.train(steps=1000, adam_rate=0.001)
    print posterior
    
    
if __name__ == "__main__":
    main()
